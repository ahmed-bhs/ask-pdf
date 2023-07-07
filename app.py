#!/usr/bin/env python

import argparse
import os
import re
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from PyPDF2.errors import PdfReadError
import json

filepath    = None

def get_pdf_text(pdf):
    try:
        text = ""
        p = 0
        pdf_reader = PdfReader(pdf)
        while len(text) <= 8000:
            page = pdf_reader.pages[p]
            text += remove_header_footer(page.extract_text())
            p = p + 1
    except PdfReadError:
        print("invalid PDF file, please check your file extension.")
        exit(2)
    t = "".join([s for s in text.splitlines(True) if s.strip("\r\n")])
    t = re.sub(r'\n\s*\n','\n', t, re.MULTILINE)
    # Remove mails
    t = re.sub(r'[\w\.-]+@[\w\.-]+', '', t)
    # Remove urls
    t = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", t)

    return re.sub(r'http\S+', '', t)

def remove_header_footer(pdf_extracted_text):
        page_format_pattern = r'([page])'
        pdf_extracted_text = pdf_extracted_text.split("\n")
        header = pdf_extracted_text[0].strip()
        footer = pdf_extracted_text[-1].strip()
        if re.search(page_format_pattern, header) or header.isnumeric():
            pdf_extracted_text = pdf_extracted_text[1:]
        if re.search(page_format_pattern, footer) or footer.isnumeric():
            pdf_extracted_text = pdf_extracted_text[:-1]
        pdf_extracted_text = "\n".join(pdf_extracted_text)
        return pdf_extracted_text

def get_text_chunks(text):
    section_pattern = r"(SECTION|RUBRIQUE) \d+: .+"
    section_headings = re.findall(section_pattern, text)
    chunks = re.split(section_pattern, text)
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    if not re.match(section_pattern, text.split("\n", 1)[0]):
        section_headings.insert(0, "Section 0: ")
    if len(section_headings) != len(chunks):
        raise ValueError("Mismatch between the number of section headings and chunks")
    for i, chunk in enumerate(chunks):
        chunk = f"{section_headings[i]}\n\n{chunk}"
        chunks[i] = chunk

    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k-0613")

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True,)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def parse_args():
    global filepath

    argsParser = argparse.ArgumentParser(usage='Parse PDF information')
    argsParser.add_argument('-f', '--file', action='store', dest='filepath', default='', help='The PDF file that will be used', required=True)
    args = argsParser.parse_args()
    filepath = os.path.normpath(args.filepath)

def main():
    load_dotenv()

    cwd = os.path.dirname(os.path.realpath(__file__))
    parse_args()

    # validate path input
    if (filepath == None):
        print('A path specification is required')
        exit(2)

    # convert relative path to absolute path
    if len(os.path.splitdrive(filepath)[0]) == 0:
        pathspec = os.path.normpath(os.path.join(cwd, filepath))

    if os.path.isdir(filepath):
        print('please specify a file arg')
        exit(2)

    raw_text = get_pdf_text(filepath)

    text_chunks = get_text_chunks(raw_text)
#     # create vector store
    vectorstore = get_vectorstore(text_chunks)
    conversation = get_conversation_chain(vectorstore)

    #https://platform.openai.com/tokenizer
    prompt_question = """
    En format JSON avec clé underscored anglais, quel est le nom,
    la coleur, le fabricant et les substances (nom et cas) du produit,
    quelles sont les mentions de danger (phrases H), les mentions additionelles (phrases EUH),
    les mentions de Conseils de prudence (phrases P), la mention d'avertissement et les pictogrammes de danger (Lister uniquement les codes sans explication) ?"
    """
    response = conversation({'question': prompt_question})

    print(response['answer'])

if __name__ == '__main__':
    main()
