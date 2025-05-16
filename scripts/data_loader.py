from typing import Union, List, Tuple
from pathlib import Path
import os
import pandas as pd
import cv2

from unstructured.partition.pdf import partition_pdf
from unstructured.partition.utils.constants import PartitionStrategy
from unstructured.documents.elements import Element

from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

from img2table.ocr import TesseractOCR
from img2table.document import Image
from img2table.tables.objects.extraction import ExtractedTable

pdfs_dir = Path(__file__).parent.parent / 'data' / 'pdfs'
figures_dir = Path(__file__).parent.parent / 'data' / 'figures'

def upload_pdf(file: Union[str, Path]) -> None:
    with open(str(pdfs_dir / file.name), 'wb') as f:
        f.write(file.getbuffer())

def extract_text_from_image(file_path: Union[str, Path], model: OllamaLLM) -> str:
    model_w_image_context = model.bind(images=[file_path])
    return model_w_image_context.invoke("Tell me what do you see in this picture.")


def separate_table_and_text_elements(elements: List[Element]) -> Tuple[List[Element], List[Element]]:
    """
    This function is to detect tables candidates where a single table may have been divided into 2 due to page breaks.
    It separates tables from extracted text elements.
    """
    texts = []
    tables = []
    group = []

    for i in range(len(elements)):
        el = elements[i]
        if el.category not in ["Table", "Image", "PageBreak"]:
            texts.append(el.text)
        elif el.category == "Table":
            group.append(el)
            # check if next element is a table or PageBreak
            # then don't do anything
            # if it is not a table or PageBreak then reset the group
            if i+1<len(elements) and elements[i+1].category in ["Table", "PageBreak"]:
                continue
            else:
                tables.append(group)
                group = []
    return texts, tables


def load_pdf(file_path: Union[str, Path], figures_dir: Union[str, Path], model: OllamaLLM) -> List[str]:
    elements = partition_pdf(
        file_path,
        strategy=PartitionStrategy.HI_RES,
        extract_image_block_types=["Image", "Table"],
        extract_image_block_output_dir=figures_dir,
        include_page_breaks=True
    )           

    text_elements, tables = separate_table_and_text_elements(elements)

    for file in os.listdir(figures_dir): 
        if "table" not in file:
            extracted_text = extract_text_from_image(Path(figures_dir, file), model)
            text_elements.append(extracted_text)

    table_df_list = extract_structured_data(tables)
    table_json_list = [df.to_json(orient="records") for df in table_df_list]
    return "\n\n".join(text_elements) + "\n\n".join(table_json_list)


def split_text(text: str) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_text(text)


def convert_to_grayscale(file_path):
    image = cv2.imread(file_path)
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply binary thresholding
    _, thresh_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)
    # Resize the image
    resized_image = cv2.resize(thresh_image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    _, im_buf_arr = cv2.imencode(".jpg", resized_image)
    img_bytes = im_buf_arr.tobytes()
    return img_bytes


def extract_structured_data(tables: List[Element]): 

    df_list = []

    # Instantiation of OCR
    ocr = TesseractOCR(n_threads=1, lang="eng")

    for group in tables:
        # Confirm here if the group tables are the same table
        headers = None 

        for el in group:
            file_path = el.metadata.image_path

            processed_img_bytes = convert_to_grayscale(file_path)
            doc = Image(src=processed_img_bytes)
            # Table extraction
            extracted_tables = doc.extract_tables(ocr=ocr,
                                implicit_rows=True,
                                implicit_columns=True,
                                borderless_tables=True,
                                min_confidence=0)
            
            assert len(extracted_tables) == 1
            table = extracted_tables[0]

            table_headers = extract_table_headers(table)
            df = extract_table_data(table)

            if headers is None and table_headers:
                headers = table_headers
                df_list.append(df)
            elif headers and table_headers:
                # check if they have the same headers
                if headers == table_headers: # then append to the previous table
                    df_list[-1] = pd.concat([df_list[-1], df], ignore_index=True)
                else:
                    # If they don't have same headers, it's a new table
                    df_list.append(df)
            elif headers and not table_headers:
                # then append to the previous table
                df.columns = headers
                df_list[-1] = pd.concat([df_list[-1], df], ignore_index=True)
            else:
                df_list.append(df)

    return df_list


def extract_table_data(table: ExtractedTable) -> pd.DataFrame:
    headers = extract_table_headers(table)
    df = table.df
    if headers:
        df.columns = headers
    return df

def has_table_headers(table: ExtractedTable) -> bool:
    if table.title:
        return True
    return False

def extract_table_headers(table: ExtractedTable) -> List[str]:
    if has_table_headers(table):
        return table.title.split("\n")
    return None
    