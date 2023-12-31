import re
import string
from itertools import product
from difflib import SequenceMatcher

def news_preprocess(text:str):
    # text = text.lower()
    text = text.replace('\n', '')
    for punc in string.punctuation:
        if punc in text:
            text = text.replace(punc, ' ' + punc)
    # translator = str.maketrans(' ', ' ', string.punctuation)
    # text = text.translate(translator)
    return text.strip()

def remove_punc(text:str):
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    return text.strip()

def longest_common_substring_word_level(str1, str2):
    # Tách từ trong hai chuỗi
    words1 = str1.split()
    words2 = str2.split()

    # Tạo một ma trận để lưu độ dài của chuỗi con chung
    matrix = [[0] * (len(words2) + 1) for _ in range(len(words1) + 1)]

    # Biến lưu độ dài của chuỗi con dài nhất và chỉ mục của nó trong words1
    max_length = 0
    end_index = 0

    for i in range(1, len(words1) + 1):
        for j in range(1, len(words2) + 1):
            if words1[i - 1] == words2[j - 1]:
                matrix[i][j] = matrix[i - 1][j - 1] + 1
                if matrix[i][j] > max_length:
                    max_length = matrix[i][j]
                    end_index = i - 1
            else:
                matrix[i][j] = 0

    # Tìm chuỗi con dài nhất bằng cách trích xuất nó từ words1
    if max_length > 0:
        start_index = end_index - max_length + 1
        longest_substring = " ".join(words1[start_index:end_index + 1])
    else:
        longest_substring = ""

    return longest_substring
    
def count_substring_occurrences(text, substrings):
    occurrences_dict = {substring: text.count(substring) for substring in substrings}
    return occurrences_dict
def make_combinations(my_list: list):
    combinations = list(product(my_list, repeat=2))

    filtered_combinations = [(x, y) for x, y in combinations if x != y]

    return(filtered_combinations)
def get_substring_by_value(input_str, start_value, end_value):
    start_index = input_str.find(start_value)
    end_index = input_str.find(end_value, start_index) + len(end_value)
    
    return input_str[start_index:end_index]
def has_title_case_word(input_str):
    words = input_str.split()
    for word in words:
        if word.istitle():
            return True
    return False
def get_title_case_position(text: str):
    for word in text.split():
        if word.istitle():
            return text[text.index(word):]
def segment_list(lst):
    result = []
    temp = [lst[0]]

    for i in range(1, len(lst)):
        if lst[i] == lst[i-1] + 1:
            temp.append(lst[i])
        else:
            result.append(temp)
            temp = [lst[i]]

    # Đảm bảo thêm list cuối cùng nếu cần
    if temp:
        result.append(temp)

    return result

def process_event(event:dict):
    output_dict = {}
    for key, values in event.items():
        for value in values:
            output_dict[value] = [key]

    return output_dict

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U0001F680-\U0001F6FF"  # Transport and Map Symbols
                               u"\U00002702-\U000027B0"  # Dingbats
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)