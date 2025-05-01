from typing import Union, List
from pathlib import Path
import os
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.utils.constants import PartitionStrategy
from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter



pdfs_dir = Path(__file__).parent.parent / 'data' / 'pdfs'
figures_dir = Path(__file__).parent.parent / 'data' / 'figures'

def upload_pdf(file: Union[str, Path]) -> None:
    with open(str(pdfs_dir / file.name), 'wb') as f:
        f.write(file.getbuffer())

def extract_text(file_path: Union[str, Path], model: OllamaLLM) -> str:
    model_w_image_context = model.bind(images=[file_path])
    return model_w_image_context.invoke("Tell me what do you see in this picture.")


def load_pdf(file_path: Union[str, Path], figures_dir: Union[str, Path], model: OllamaLLM) -> List[str]:
    elements = partition_pdf(
        file_path,
        strategy=PartitionStrategy.HI_RES,
        extract_image_block_types=["Image", "Table"],
        extract_image_block_output_dir=figures_dir
    )

    text_elements = [element.text for element in elements if element.category not in ["Image", "Table"]]

    for file in os.listdir(figures_dir): 
        extracted_text = extract_text(Path(figures_dir, file), model)
        text_elements.append(extracted_text)
    return "\n\n".join(text_elements)

def split_text(text: str) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_text(text)

