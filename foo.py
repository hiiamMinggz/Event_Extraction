from utils import *
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, BertTokenizer, BertModel
from transformers import pipeline
from collections import Counter
from underthesea import sent_tokenize
import torch.nn.functional as F

class Event_Extraction():

    def __init__(self, news) -> None:
        print("Khởi tạo")
        self.news = news
        self.title , self.content, self.avg_len = self.news_segmentation()
        self.ner_model = AutoModelForTokenClassification.from_pretrained('model/NER/')
        self.ner_tokenizer = AutoTokenizer.from_pretrained('model/NER/')
        self.ner_vietnamese = pipeline("ner", model=self.ner_model, tokenizer=self.ner_tokenizer)
        
        self.bert_tokenizer = BertTokenizer.from_pretrained('model/BERT/')
        self.bert_model = BertModel.from_pretrained('model/BERT/') 
        print('Done')
    def news_segmentation(self):
        self.title = news_preprocess(self.news['title'])
        list_sent = sent_tokenize(self.news['content'])
        self.content = [news_preprocess(sent) for sent in list_sent]
        # content = news_preprocess(self.news['content'])
        # paragraphs = sent_tokenize(content)
        sum = 0
        for text in list_sent:
            sum += len(text)
        self.avg_len = sum/len(list_sent)
        # self.content = paragraphs
        return self.title, self.content, self.avg_len
    
    def _print(self):
        print(self.title)
        sum = 0
        for text in self.content:
            print(text)
            sum += len(text)
            print('_______')
        print(sum/len(self.content))
    def call_ner_model(self, text:str):
        ner_result = self.ner_vietnamese(text)
        return ner_result
    def call_bert_model(self, text: str):
        text_tokenized = self.bert_tokenizer(text)
        text_tokenized_ids = torch.tensor([text_tokenized['input_ids']])
        pooler_output = self.bert_model(text_tokenized_ids).pooler_output
        return pooler_output
    def process_ner_result(self, ner_result: list):
        indexs = []
        ner_in_text = []
        for ner in ner_result:
            indexs.append(ner['index'])
        segmented_indexs = segment_list(indexs)
        for ele in segmented_indexs:
            text = ''
            for i in range(len(ele)):
                text += ner_result.pop(0)['word'] + ' '
            # print(text)
            ner_in_text.append(text)
        result = []
        for ele in ner_in_text:
            result.append(ele.replace(' ##', '').rstrip())
        result = list(dict.fromkeys(result))
        return result
    def static_useful_word(self):
        substrings = []
        # print(self.title)
        for text in self.content:
            text = remove_punc(text)
            common_text = longest_common_substring_word_level(self.title, text)
            # print(text)
            # print(common_text)
            # print('--------------------')
            if common_text != '':
                substrings.append(common_text)
        substrings = list(dict.fromkeys(substrings))
        # print(substrings)
        whole_text = self.title + ''.join(self.content)
        # print(len(self.content))
        occurrences_dict = count_substring_occurrences(whole_text, substrings)
        confirm_occurrences = [key for key in occurrences_dict.keys() if occurrences_dict[key] >= 3]
        # print(occurrences_dict)
        useful_words = []
        for words in confirm_occurrences:
            if has_title_case_word(words):
                useful_words.append(words)
        if len(useful_words) < 2:
            useful_words = confirm_occurrences
        # print(useful_words)
        return useful_words
    def extract_main_event_candidate(self):
        useful_words = self.static_useful_word()
        combinations = make_combinations(useful_words)
        candidates = []
        for combi in combinations:
            head = combi[0]
            tail = combi[1]
            try :
                candidate = get_substring_by_value(self.title, head, tail)
            except Exception as e :
                continue
            if candidate != '':
                # print(head)
                # print(tail)
                # print(candidate)
                # print('-------------')
                candidates.append(candidate)
        candidates = list(dict.fromkeys(candidates))
        filtered_candidates = []
        for candidate in candidates:
            filtered_candidates.append(get_title_case_position(candidate))
        filtered_candidates = list(filter(lambda x: x is not None, filtered_candidates))
        filtered_candidates = list(dict.fromkeys(filtered_candidates))
        return filtered_candidates
    def extract_sub_event_candidate(self):
        tail_content = self.content[int(len(self.content)*0.6):]
        # print(tail_content)
        ner_results = []
        for i in range (len(tail_content)):
            if self.call_ner_model(tail_content[i]):
                ner_results.append(self.call_ner_model(tail_content[i]))
        flattened_ner_result = [item for sublist in ner_results for item in sublist]
        ner_in_text = self.process_ner_result(flattened_ner_result)
        # print(ner_in_text)
        useful_words = self.static_useful_word()
        # print(useful_words)
        combinations = []
        for item1 in ner_in_text :
            for item2 in useful_words:
                combinations.append((item1, item2))
                combinations.append((item2, item1))
        sub_event_candidates = []
        for combi in combinations:
            head = combi[0]
            tail = combi[1]
            try :
                candidate = get_substring_by_value(' '.join(tail_content), head, tail)
            except Exception as e :
                continue
            if candidate != '' and len(candidate) < self.avg_len:
                # print(head)
                # print(tail)
                # print(candidate)
                # print('-------------')
                sub_event_candidates.append(candidate)
        sub_event_candidates = list(dict.fromkeys(sub_event_candidates)) 
        
        return sub_event_candidates
    def ranking_event(self):
        main_event_candidates = self.extract_main_event_candidate()
        tmp_1 = [i for i in range(len(main_event_candidates))]
        sub_event_candidates = self.extract_sub_event_candidate()
        tmp_2 = [i for i in range(len(sub_event_candidates))]
        main_event_dict = dict(zip(main_event_candidates, tmp_1))
        sub_event_dict = dict(zip(sub_event_candidates, tmp_2))
        title_vectorized = self.call_bert_model(self.title)
             
        for key in main_event_dict:
            key_vectorized = self.call_bert_model(str(key))
            cosine_similarity = F.cosine_similarity(title_vectorized, key_vectorized)
            main_event_dict.update({key:cosine_similarity.item()})
            
        for key in sub_event_dict:
            key_vectorized = self.call_bert_model(str(key))
            cosine_similarity = F.cosine_similarity(title_vectorized, key_vectorized)
            sub_event_dict.update({key:cosine_similarity.item()})
        sorted_main_event_dict = dict(sorted(main_event_dict.items(), key=lambda x:x[1], reverse=True))
        sorted_sub_event_dict = dict(sorted(sub_event_dict.items(), key=lambda x:x[1], reverse=True))
        
        # print((sorted_main_event_dict))
        # print(sorted_sub_event_dict)
        
        
        return {self.title : [list(sorted_main_event_dict.keys())[0], list(sorted_sub_event_dict.keys())[0]]}
        # return {self.title : [list(sorted_main_event_dict.items())[0], list(sorted_sub_event_dict.items())[0]]}
        
    def handle_relevant_event(self, exist_event: dict):
        current_event = self.ranking_event()
        # print(current_event)
        processed_current_event = process_event(current_event)
        
        threshold = 0.5
        if not exist_event :
            exist_event = processed_current_event
        print(processed_current_event)
        print(exist_event)
        for key_exist in exist_event.keys():
            for key_current in processed_current_event.keys():
                sim_score = similar(key_exist, key_current)
                if sim_score > threshold:
                    exist_event[key_exist] += processed_current_event[key_current]
        return exist_event
        
if __name__ == '__main__':
    news = {
        'title' : 'Lionel Messi lần thứ 8 giành Quả bóng vàng, lập kỷ lục vĩ đại',
        'content' : 'Messi đã vượt qua Haaland để trở thành chủ nhân của danh hiệu Quả bóng vàng 2023. Đây là lần thứ 8 El Pulga giành giải thưởng này. Một con số cho thấy sự vĩ đại của ngôi sao người Argentina. Không nằm ngoài dự đoán, Lionel Messi đã giành giải Quả bóng vàng 2023 của tạp chí France Football. Buổi lễ trao giải thưởng cao quý này diễn ra ở Nhà hát Chatelet (Paris, Pháp) vào đêm qua. Điều đặc biệt, người xướng tên và trao giải cho Messi là David Beckham, đồng sở hữu CLB Inter Miami. Chính Beckham là người có công lớn nhất đưa El Pulga sang Mỹ thi đấu, mở ra chương mới đầy thú vị trong sự nghiệp cầu thủ này. Đây là danh hiệu Quả bóng vàng thứ 8 trong sự nghiệp của Messi. Trước đó, siêu sao số 10 đã đăng quang vào các năm 2009, 2010, 2011, 2012, 2015, 2019 và 2021. Cầu thủ này đã nối dài kỷ lục vĩ đại khi trở thành cầu thủ giành nhiều Quả bóng vàng nhất trong sự nghiệp, vượt xa người xếp thứ hai là C.Ronaldo (5 lần). Thực tế, việc Messi giành giải Quả bóng vàng 2023 đã được dự đoán từ nhiều ngày trước buổi lễ. Siêu sao người Argentina đã trải qua năm thi đấu vô cùng thành công. Đặc biệt, Messi có công lớn giúp đội tuyển Argentina giành chức vô địch World Cup 2022 trên đất Qatar. Đó là giải đấu El Pulga tỏa sáng rực rỡ khi đóng góp tới 7 bàn thắng (trong đó có cú đúp trong trận chung kết với Pháp) để đưa Argentina lên đỉnh thế giới lần đầu tiên kể từ năm 1986. Anh đã giành giải Cầu thủ xuất sắc nhất World Cup 2022 sau màn trình diễn siêu hạng. Chính điều này giúp Messi được đánh giá cao hơn so với Erling Haaland trong cuộc đua Quả bóng vàng. Tiền đạo người Na Uy cũng trải qua một năm đầy ắp thành công cùng Man City với cú ăn ba lịch sử. Mặc dù vậy, theo đánh giá của nhiều chuyên gia, những danh hiệu của Haaland đã bị lu mờ bởi chức vô địch World Cup của Messi. Dù sao, Haaland cũng được an ủi khi nhận được danh hiệu Quả bóng bạc và giải Gerd Muller Trophy (dành cho tiền đạo xuất sắc nhất) trong buổi lễ trao giải. Trong khi đó, Kylian Mbappe giành Quả bóng đồng. Phát biểu trên bục nhận giải, Messi cho biết: "Thật vui khi được đứng ở đây một lần nữa và tận hưởng khoảnh khắc này. Cảm ơn tới tất cả. Tôi muốn chia sẻ danh hiệu này với các đồng đội vì những đóng góp mang về thành công lớn cho đội tuyển Argentina. Tôi muốn cảm ơn ban huấn luyện cùng tất cả những người đã làm nên chiến tích vĩ đại của Argentina. Tôi không thể tưởng tượng được những gì mình đã đạt được trong sự nghiệp. Tôi may mắn khi trở thành một phần trong đội bóng xuất sắc nhất lịch sử. Chúng tôi đã giành hai chức vô địch Copa America và World Cup liên tiếp sau những thời khắc vô cùng khó khăn". Nói về Messi, Beckham chia sẻ: "Thật đặc biệt và tự hào khi nói rằng Messi là cầu thủ của Inter Miami và sinh sống ở thành phố Miami xinh đẹp. Anh ấy sẽ ăn mừng danh hiệu này cùng các đồng đội và gia đình theo cách của riêng mình. Chúng tôi cũng sẽ ăn mừng danh hiệu này theo cách của Miami. Tôi chắc chắc mọi người sẽ có buổi tiệc vui vẻ". Trong khi đó, danh hiệu Quả bóng vàng dành cho nữ cầu thủ thuộc về Aitana Bonmati. Tiền vệ 25 tuổi này đã tỏa sáng rực rỡ giúp Barcelona vô địch Champions League nữ và đưa đội tuyển Tây Ban Nha giành chức vô địch World Cup nữ 2023. Danh hiệu Thủ môn xuất sắc nhất (Lev Yashin Trophy) thuộc về Emiliano Martinez. Cầu thủ này là điểm tựa chắc chắn giúp đội tuyển Argentina vô địch World Cup 2022. Giải thưởng Cầu thủ trẻ xuất sắc nhất (Kopa Trophy) thuộc về Jude Bellingham. Cầu thủ trẻ người Anh thi đấu vô cùng ấn tượng trong màu áo Real Madrid ở mùa giải này.'
    }
    
    news_2 = {
        'title' : 'Messi đoạt Quả Bóng Vàng thứ tám',
        'content' : 'Ở tuổi 36, Lionel Messi vượt qua Erling Haaland và Kylian Mbappe để giành Quả Bóng Vàng 2023, nhờ chức vô địch World Cup đầu tiên trong sự nghiệp.Chủ tịch CLB Inter Miami David Beckham xướng tên Messi, trong lễ trao giải thưởng cá nhân danh giá trong giới túc cầu, do tạp chí France Football tổ chức ở nhà hát Chatelet, thành phố Paris tối 30/10. Siêu sao Argentina không lộ nhiều cảm xúc khi nghe tên, và Haaland cũng vậy, trong khi Mbappe khẽ mỉm cười trong một thoáng.Haaland nhận Bóng Bạc, còn Mbappe đạt Bóng Đồng. "Ba quả bóng này đều đặc biệt, nhưng điều quan trọng nhất vẫn là các danh hiệu đồng đội", Messi nói khi nhận Bóng Vàng.Còn khi được hỏi liệu Bóng Vàng thứ tám có đặc biệt hơn bảy giải thưởng trước đó, Messi nói: "Cả tám giải đều quan trọng vì những lý do khác nhau".Messi lập kỷ lục giành tám Quả Bóng Vàng, các năm 2009, 2010, 2011, 2012, 2015, 2019, 2021 và 2023. Anh bỏ xa cầu thủ đứng sau với năm danh hiệu là Cristiano Ronaldo. Do Bóng Vàng chỉ trao cho những cầu thủ châu Âu cho đến năm 1994, những danh thủ Nam Mỹ như Pele hay Diego Maradona chưa từng giành giải. Vì thế France Football từng trao danh dự bảy Bóng Vàng cho Pele và hai giải cho Maradona.Sau lời cảm ơn gia đình và những người đã bình chọn cho anh, Messi cũng dành lời tri ân người thầy quá cố Maradona, bởi hôm nay kỷ niệm 63 năm ngày sinh của ông. "Giải thưởng này là món quà tuyệt vời với mọi người Argentina", anh nói. "Tôi muốn nhắc tới người cuối cùng, đó là Maradona. Chúc mừng sinh nhật thầy, và không có nơi nào tốt hơn ở đây để tôi chúc thầy, bởi có rất nhiều cầu thủ đang hiện diện, cùng quả bóng này. Vì thế, giải thưởng này là của thầy, và của mọi người Argentina".Maradona dẫn đội tuyển Argentina thời 2008-2010, trao băng thủ quân cho Messi lần đầu. Ông đạt tỷ lệ thắng 75% cùng Argentina, cao nhất lịch sử đội tuyển, trong những HLV đã làm việc 10 trận trở lên. Tuy nhiên, thầy trò Maradona phải dừng bước ở tứ kết World Cup 2010.Sau bốn kỳ dự World Cup với vị trí cao nhất là á quân năm 2014, Messi lần đầu vô địch năm 2022 ở Qatar. Anh ghi bảy bàn, là cầu thủ hay nhất giải, sau khi Argentina thắng Pháp ở loạt đá luân lưu chung kết ngày 18/12/2022. Tại giải đó, Messi ghi bàn ở tất cả vòng đấu.Mbappe là Vua phá lưới World Cup với tám bàn, trong đó có hat-trick trong trận chung kết, nên lần đầu vào Top 3 Quả Bóng Vàng. Haaland giúp Man City đoạt cú ăn ba, là Vua phá lưới Ngoại hạng Anh lẫn Champions League, những thành tích khá giống người thắng giải năm ngoái là Karim Benzema. Tuy nhiên, World Cup là giải lớn nhất hành tinh, diễn ra bốn năm một lần, nên thường ảnh hưởng lớn tới các cuộc bình chọn Quả Bóng Vàng. Messi nói rằng anh tin Mbappe và Haaland sẽ giành Bóng Vàng trong những năm tới.Trong lúc Messi nhận giải và phát biểu, máy quay có lúc chiếu đến các con trai của anh, và chúng trông có vẻ chán nản. Con cả Thiago chống cằm, con thứ Mateo ngáp dài, còn con út Ciro thậm chí nằm ra trên ghế.Một lúc sau, gia đình được mời lên sân khấu chia vui cùng Messi. Các cậu bé đập tay với cựu tiền đạo Bờ Biển Ngà kiêm người dẫn chương trình Didier Drogba, rồi đứng cạnh bố. Buổi lễ kết thúc với hình ảnh Messi bên gia đình, và sau khi Beckham hứa sẽ tổ chức một tiệc mừng thật vui ở Miami. Trên màn hình lớn cũng xuất hiện dòng chữ: "Messi là vô cực", với biểu tượng số 8 nằm ngang.Messi là cầu thủ đầu tiên nhận Quả Bóng Vàng khi thuộc biên chế một CLB ngoài châu Âu. Anh rời PSG, chuyển sang khoác áo Miami từ tháng 7/2023, giúp CLB Mỹ đoạt danh hiệu đầu tiên trong lịch sử là Leagues Cup. Messi đã kết thúc mùa giải với Miami, và có thể không chơi trận nào ở cấp CLB trong hơn ba tháng tới.Top 10 Quả Bóng Vàng 2023: 1- Lionel Messi, 2- Erling Haaland, 3- Kylian Mbappe, 4- Kevin de Bruyne, 5- Rodri, 6- Vinicius Junior, 7- Julian Alvarez, 8- Victor Osimhen, 9- Bernardo Silva, 10- Luka Modric.'
    }
    
    my_event_module = Event_Extraction(news_2)
    # my_event_module._print()
    # ner_result = my_event_module.call_ner_model('Messi giành Quả bóng vàng. Trong khi đó, danh hiệu Quả bóng vàng dành cho nữ cầu thủ thuộc về Aitana Bonmati.')
    # print(ner_result)
    # print(my_event_module.process_ner_result(ner_result))
    # print(my_event_module.static_useful_word())
    # list_main_event = my_event_module.extract_main_event_candidate()
    # list_sub_event = my_event_module.extract_sub_event_candidate()
    # print(list_main_event)
    # print(list_sub_event)
    # print(my_event_module.ranking_event())
    print(my_event_module.handle_relevant_event({'Messi lần thứ 8 giành Quả bóng vàng': ['Lionel Messi lần thứ 8 giành Quả bóng vàng , lập kỷ lục vĩ đại'], 'Quả bóng vàng dành cho nữ cầu thủ thuộc về Aitana Bonmati': ['Lionel Messi lần thứ 8 giành Quả bóng vàng , lập kỷ lục vĩ đại']}))