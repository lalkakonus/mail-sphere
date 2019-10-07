from ..dataloader import DataLoader

def process():
    dataloader = DataLoader()
    doc_len = 0
    for doc in dataloader.processed_content():
        doc_len += len(doc["body"])
    return doc_len / dataloader.processed_content_len

