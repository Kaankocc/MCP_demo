# app/utils.py
import ast

def parse_query_string(query_string):
    input_dict = ast.literal_eval(query_string)
    return {
        "content_string_query": input_dict.get("content_string_query", ""),
        "industry_filter": input_dict.get("industry_filter", []) or [],
        "takeaways_filter": input_dict.get("takeaways_filter", []) or [],
    }


def format_documents(documents):
    formatted_documents = []
    for doc in documents['matches']:
        formatted_documents.append({
            "Passage": doc['metadata']['content'],
            "Interviewee": doc['metadata']['Interviewee'],
            "Industry Sectors": doc['metadata']['Industry Sectors'],
            "Takeaways": doc['metadata']['Takeaways'],
            "Source": doc['metadata']['Source'],
        })
    return formatted_documents
