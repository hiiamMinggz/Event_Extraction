from utils import *
from hermetrics.levenshtein import Levenshtein

class Event_Extraction():

    def __init__(self, news) -> None:
        self.news = news
        self.title , self.content = self.news_segmentation()
    def news_segmentation(self):
        self.title = news_preprocess(self.news['title'])
        content = news_preprocess(self.news['content'])
        paragraphs = content.split('.')
        self.content = paragraphs
        return self.title, self.content
    
    def _print(self):
        print(self.title)
        for text in self.content:
            print(text)
    def extract_event_candidates(self):
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
        confirm_occurrences = [key for key in occurrences_dict.keys() if occurrences_dict[key] >= 2]
        # print(occurrences_dict)
        useful_words = []
        for words in confirm_occurrences:
            if has_title_case_word(words):
                useful_words.append(words)
        if len(useful_words) < 2:
            useful_words = confirm_occurrences
        # print(useful_words)
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
        return filtered_candidates
    def event_extraction(self):
        lev = Levenshtein()
        candidates = self.extract_event_candidates()
        tmp_dict = {}
        for candi in candidates:
            tmp_dict.update({candi: lev.similarity(candi, self.title)})
        result = sorted(tmp_dict, reverse=True)
        return result
if __name__ == '__main__':
    # news ={
    #     'title' : 'Nga bất ngờ rút khỏi thỏa thuận giải trừ hạt nhân với Nhật Bản' ,
    #     'content' : '(Dân trí) - Chính phủ Nga bất ngờ đình chỉ thỏa thuận hợp tác với Nhật Bản về giải trừ vũ khí hạt nhân của Moscow, theo thông tin từ một tài liệu mà Điện Kremlin công bố vào tối 9/11.\n\nTài liệu được đăng trên Cổng thông tin trực tuyến chính thức của chính phủ Nga cho thấy, Thủ tướng Mikhail Mishustin đã ký sắc lệnh đình chỉ thỏa thuận này từ ngày 7/11.\n\n"Chấm dứt thỏa thuận giữa chính phủ Nga và chính phủ Nhật Bản ký ngày 13/10/1993 tại Tokyo về hợp tác loại bỏ vũ khí hạt nhân ở Nga và thành lập ủy ban hợp tác cho mục đích này", tài liệu nêu rõ.\n\n Phía Moscow không đưa ra lý do cho quyết định này, nhưng cho biết Bộ Ngoại giao Nga đã được chỉ đạo thông báo cho phía Nhật Bản về động thái này.\n\nTrong khi đó, đài NHK của Nhật Bản dẫn lời Chánh văn phòng Nội các nước này Matsuno Hirokazu cho biết, Tokyo không nhận được thông báo nào về quyết định của Nga, đồng thời đơn phương bày tỏ sự đáng tiếc về động thái này.\n\n"Chúng tôi cảm thấy đáng tiếc về quyết định của Nga, được đơn phương đưa ra mà không thông báo cho Nhật Bản. Chúng tôi dự định xác nhận thông tin cụ thể thông qua kênh ngoại giao", Chánh văn phòng Nội các Nhật Bản nói tại cuộc họp báo trong ngày 10/11.'
    # }
    # news = {
    #     'title' : "Lionel Messi lần thứ 8 giành Quả bóng vàng, lập kỷ lục vĩ đại",
    #     'content' : "(Dân trí) - Messi đã vượt qua Haaland để trở thành chủ nhân của danh hiệu Quả bóng vàng 2023. Đây là lần thứ 8 El Pulga giành giải thưởng này. Một con số cho thấy sự vĩ đại của ngôi sao người Argentina.\nKhông nằm ngoài dự đoán, Lionel Messi đã giành giải Quả bóng vàng 2023 của tạp chí France Football. Buổi lễ trao giải thưởng cao quý này diễn ra ở Nhà hát Chatelet (Paris, Pháp) vào đêm qua.\n\n Điều đặc biệt, người xướng tên và trao giải cho Messi là David Beckham, đồng sở hữu CLB Inter Miami. Chính Beckham là người có công lớn nhất đưa El Pulga sang Mỹ thi đấu, mở ra chương mới đầy thú vị trong sự nghiệp cầu thủ này. \n\n Đây là danh hiệu Quả bóng vàng thứ 8 trong sự nghiệp của Messi. Trước đó, siêu sao số 10 đã đăng quang vào các năm 2009, 2010, 2011, 2012, 2015, 2019 và 2021."
    # }
    # news ={
    #     'title' : 'Thủ đô Ukraine nghi bị tập kích bằng tên lửa' ,
    #     'content' : '(Dân trí) - Thị trưởng Kiev Vitali Klitschko cho biết, đêm 10/11, rạng sáng 11/11, khu vực Kiev bị tấn công, nhiều tiếng nổ lớn đã vang lên.\n\n"Nhiều tiếng nổ lớn dội lên ở phía tả ngạn (sông Dnieper) của thủ đô. Theo thông tin ban đầu, hệ thống phòng không đã được kích hoạt để đánh chặn tên lửa đạn đạo", ông Klitschko viết trên Telegram.\n\nHiện chưa có thông tin về thương vong, thiệt hại sau vụ tập kích.\n\nĐây là lần đầu tiên Kiev bị tập kích kể từ cuối tháng 9. Còi báo động chỉ vang lên vài phút trước khi xuất hiện những tiếng nổ lớn. Chính quyền thành phố đã kêu gọi người dân xuống hầm trú ẩn.\n\nChính quyền quân sự Kiev sau đó cho biết, hệ thống phòng không của quân đội đã đánh chặn toàn bộ máy bay không người lái (UAV) của Nga. Ukraine cho rằng, cuộc tập kích là một phần trong những hoạt động của Nga nhằm xác định và phá hủy hệ thống phòng không ở thủ đô Kiev.\n\nTheo Không quân Ukraine, trong đêm 10/11 và rạng sáng 11/11, các hệ thống phòng không của họ đã phá hủy 19 trong tổng số 31 UAV Shahed của Nga cùng với một số tên lửa. Trong đó, một tên lửa Kh-31 bắn từ Biển Đen, một tên lửa chống hạm P-800 Oniks bắn từ Crimea và một tên lửa phòng không dẫn đường S-300 từ tỉnh Belgorod của Nga.\n\nCuộc tập kích diễn ra trong bối cảnh Ukraine và phương Tây cho rằng Nga đã chuẩn bị một kho vũ khí rất lớn để sẵn sàng cho những cuộc tấn công vào mùa đông nhằm phá hủy hạ tầng năng lượng của Ukraine.\n\nÔng Vadym Skibitskyi, đại diện Tổng cục Tình báo của Bộ Quốc phòng Ukraine (HUR) cho biết, chỉ trong tháng 10 Nga đã sản xuất tới 115 tên lửa chính xác cao để tăng cường cho kho dự trữ mùa đông.'
    # }
    news = {
        'title' : 'Messi lần thứ 8 giành Quả bóng vàng',
        'content' : 'Lionel Messi có lần thứ 8 được France Football vinh danh ở hạng mục Quả bóng vàng.\n\n Rạng sáng 31.10 (giờ Việt Nam), gala trao giải Quả bóng vàng 2023 đã được tạp chí France Football tổ chức tại nhà hát Chatelet (Paris, Pháp). Quả bóng vàng 2023 cùng một số giải cá nhân khác dành cho những cầu thủ xuất sắc nhất mùa giải 2022-2023.\n\nTrước khi kết quả được công bố, giới bình luận cũng như người hâm mộ tập trung vào 2 cái tên Lionel Messi và Erling Haaland ở cuộc đua dành cho các cầu thủ nam. Thông tin rò rỉ tiết lộ, người chiến thắng là Messi.\n\n Và thực tế đúng như vậy, khi đội trưởng đội tuyển Argentina nhận số điểm bình chọn cao nhất, vượt qua Haaland và Kylian Mbappe.\n\n Đây là lần thứ 8, Messi giành Quả bóng vàng, nới rộng khoảng cách với đối thủ lớn nhất trong sự nghiệp của anh là Cristiano Ronaldo lên thành 3 danh hiệu. Siêu sao người Bồ Đào Nha có 5 Quả bóng vàng.\n\n Trước chiến thắng này, Messi đã được vinh danh vào các năm 2009, 2010, 2011, 2012, 2015, 2019 và 2021.\n\n Đáng chú ý, năm ngoái, Messi thậm chí còn không lọt danh sách đề cử vì mùa giải không thành công với Paris St Germain. Ở mùa giải 2021-2022, đội bóng Pháp không thành công trên đấu trường châu Âu, nên chiến thắng cuối cùng thuộc về Karim Benzema với mùa giải xuất sắc cùng Real Madrid.\n\n Messi cũng không có mùa giải cuối cùng với PSG quá tốt, nhưng nhờ màn trình diễn xuất sắc và chiến thắng cùng đội tuyển Argentina ở World Cup 2022, El Pulga xứng đáng được vinh danh. \n\n Ở giải đấu tại Qatar cuối năm ngoái, vai trò của Messi nổi bật khi dẫn dắt “La Albiceleste” trên hành trình chinh phục chiếc cúp vàng. Anh đã ghi bàn ở 6/7 trận, in dấu giày ở 10/15 bàn thắng của cả đội (7 bàn, 3 kiến tạo).\n\n Đối thủ lớn nhất của Messi trong cuộc đua năm nay là Haaland, chân sút người Na Uy đã ghi 52 bàn thắng trên mọi đấu trường cho Man City ở mùa giải 2022-2023.'
    }
    event = Event_Extraction(news=news)
    # event._print()
    print(event.event_extraction())
    