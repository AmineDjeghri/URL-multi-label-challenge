from urllib.parse import urlparse
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score

def url_parse(url):
    parse_result = urlparse(url)
    result = [parse_result.scheme, parse_result.netloc, parse_result.path, parse_result.params, parse_result.query,
              parse_result.fragment]
    return result


def split_netloc(netloc: str):
    splited_netloc = netloc.rsplit('.', 2)
    if len(splited_netloc) == 2:
        splited_netloc.insert(0, "www")
    return splited_netloc


def list_string_to_int(liste):
    try:
        return [int(x) for x in liste]
    except:
        print(liste)

def display_score(y_test, predictions):
    print("Accuracy :",metrics.accuracy_score(y_test, predictions))
    print("Hamming loss ",metrics.hamming_loss(y_test,predictions))
    precision = precision_score(y_test, predictions, average='micro')
    recall = recall_score(y_test, predictions, average='micro')
    f1 = f1_score(y_test, predictions, average='micro')

    print("\nMicro-average quality numbers")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))
    precision = precision_score(y_test, predictions, average='macro')
    recall = recall_score(y_test, predictions, average='macro')
    f1 = f1_score(y_test, predictions, average='macro')

    print("\nMacro-average quality numbers")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))
    print("\nClassification Report")
    print (metrics.classification_report(y_test, predictions))