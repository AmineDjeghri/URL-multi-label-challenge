from urllib.parse import urlparse


def url_parse(url):
    parse_result = urlparse(url)
    result = [parse_result.scheme, parse_result.netloc, parse_result.path, parse_result.params, parse_result.query,
              parse_result.fragment]
    return result

def split_netloc(netloc:str):
    splited_netloc = netloc.rsplit('.', 2)
    if len(splited_netloc) == 2:
        splited_netloc.insert(0, "www")
    return splited_netloc




