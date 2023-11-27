from utils import *
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, BertTokenizer, BertModel
from transformers import pipeline
from collections import Counter
from underthesea import sent_tokenize
import torch.nn.functional as F

class Post_Event_Extraction():

    def __init__(self, post) -> None:
        print("Khởi tạo")
        self.post = remove_emoji(post)
                
        self.bert_tokenizer = BertTokenizer.from_pretrained('model/BERT/')
        self.bert_model = BertModel.from_pretrained('model/BERT/')
        print("Done")
    def _print(self):
        print(self.post)
    def post_event_extraction(self):
        if '\n' in self.post:
            paras = self.post.split('\n')
            if len(sent_tokenize(paras[0])) < 2 and len(paras) > 4:
                return {self.post: [paras[0]]}
            return {self.post: ['None']}
        else:
            return {self.post: ['None']}
    def handle_relevant_event(self, exist_event: dict):
        current_event = self.post_event_extraction()
        # print(current_event)
        processed_current_event = process_event(current_event)
        
        threshold = 0.5
        if not exist_event :
            exist_event = processed_current_event
            return exist_event
        # print(processed_current_event)
        # print(exist_event)
        for key_exist in exist_event.keys():
            for key_current in processed_current_event.keys():
                # print(news_preprocess(key_current))
                sim_score = similar(news_preprocess(key_exist).lower(), news_preprocess(key_current).lower())
                print(sim_score)
                if sim_score > threshold:
                    exist_event[key_exist] += processed_current_event[key_current]
        return exist_event 
if __name__ == '__main__':
    post = """Hòa thượng Thích Tuệ Sỹ đã viên tịch vào lúc 16:00 pm hôm 24/11/2023.
Hòa thượng (HT) đã từng ở tù cộng sản hai lần, tổng cộng 17 năm tù. HT từng là giáo sư của đại học Vạn Hạnh - Sài Gòn. Năm 1984, HT bị nhà cầm quyền Việt Nam bắt giam, đến năm 1988 bị tuyên án tử hình với tội danh "Hoạt động nhằm lật đổ chính quyền nhân dân", cùng với HT Thích Trí Siêu (Lê Mạnh Thát). Sau đó, do sự vận động của các tổ chức nhân quyền, bản án được giảm xuống thành chung thân. 
Ngày 1/9/1998, HT được trả tự do từ trại Ba Sao-Nam Hà. Trước khi thả, nhà cầm quyền yêu cầu ông ký vào lá đơn xin ân xá. HT trả lời: “Không ai có quyền xét xử tôi, không ai có quyền ân xá tôi!”. Công an nói không viết đơn thì không thả, HT không viết và tuyệt thực. Nhà cầm quyền đã phải phóng thích ngài sau mười ngày tuyệt thực. 
Được trả tự do nhưng HT vẫn bị quản thúc cho đến bây giờ. HT đã dành cả đời để tu tập và đấu tranh cho sự từ bi và công bằng cho đất nước. 
Đạo là đời, đời là đạo. Tâm tu hành nhưng thân xác vẫn phải lăn trải vào đời để hành đạo, đây mới là con đường chân chính, tu mà không hành thì chưa phải là tu, hành mà không hiểu về giáo lý thì chỉ là tu mù. Hy vọng HT sẽ là tấm gương cho những vị tu sĩ, cư sĩ ngày nay hiểu được tu hành chân chính là thế nào, với một xã hội đầy nhiễu nhương, sự tu hành đầy chông chênh, gập ghềnh, dễ bị cám dỗ và lạc lối. 
Kính tiễn Hòa Thượng về miền an lạc.
    """
    post_2 = """NGHI VẤN: ĐẠI BIỂU QUỐC HỘI BUÔN BÁN CAMERA GIÁM SÁT
Sau hàng loạt cách thức “hành” dân chúng, thì nay, Quốc hội lại vừa phát minh ra một ý tưởng chứng tỏ sự “đỉnh cao trí tuệ”, đó chính là gắn camera giám sát hành trình vào phương tiện mô tô, xe máy.
Làn sóng phẫn nộ trong dân lại bùng lên, với những tranh cãi nảy lửa. Đi kèm với sự bức xúc, ý tưởng điên rồ này có thể tạo ra vô vàn hệ lụy cho dân chúng.
Chúng ta vẫn thường thấy xuất hiện trên các phương tiện truyền thống đại chúng, hay chỉ cần bước ra đường, hình ảnh quan chức, lãnh đạo cậy quyền hành mà giẫm đạp lên luật pháp, coi thường dân đen là chuyện như cơm bữa. Nếu muốn gắn camera giám sát, thiết nghĩ, những đối tượng này cần trước áp dụng trước hết.
Bộ máy quản lý hiện đã rất lớn và tiêu tốn biết bao tiền của ngân sách. Việc hàng triệu camera giám sát được đưa vào vận hành sẽ kéo theo sự gia tăng nhân lực ngành công an để xử lý các dữ liệu, điều đó có cần thiết và mang lại hiệu quả gì tốt đẹp hay không? Hay đơn thuần chỉ là công cụ để kiểm soát người dân một cách dễ dàng hơn?
Một bộ phận không nhỏ đặt ra nghi vấn về “lợi ích nhóm”, xuất phát từ mục đích bán các thiết bị giám sát cho hàng trăm triệu phương tiện tham gia giao thông nhằm thu về một khoản lợi lớn.
Và cuối cùng, dù cho ý nghĩa, mục đích được đưa ra to lớn cỡ nào, thì người chịu thiệt hại vẫn là dân. Ngoài khoản tiền khi phải lắp đặt thiết bị mới, thì khi xảy ra hỏng hóc, hay mất mát, họ cũng đối diện với rủi ro bị phạt.
Trong tình cảnh kinh tế suy thoái, đời sống người dân vô cùng khó khăn, mọi quyết định ảnh hưởng đến quyền lợi của dân không nên được thông qua.
Linh Linh
    """
    post_3 = """Trong chế độ này, đụng đâu cũng thấy tham nhũng, cứ mỗi lần khui ra là nhân dân lại choáng váng vì toàn … đại án!
"""
    post_4 = """HOÀ THƯỢNG THÍCH TUỆ SỸ ĐÃ VIÊN TỊCH!
GNO - Hòa thượng Thích Tuệ Sỹ, vị giáo phẩm uyên bác, dịch giả của nhiều bộ kinh, luật, luận quan trọng; tác giả của các công trình nghiên cứu Phật học giá trị, sau thời gian bệnh duyên đã viên tịch tại chùa Phật  n (tỉnh Đồng Nai) vào 16 giờ chiều, ngày 24-11-2023 (12-10-Quý Mão).
Hòa thượng Thích Tuệ Sỹ, pháp húy Nguyên Chứng, sinh năm 1943 tại Paksé (Lào), nguyên quán tại tỉnh Quảng Bình.
Hòa thượng Thích Tuệ Sỹ là đệ tử của Đại lão Hòa thượng Thích Trí Thủ (1909-1984), Đệ nhất Chủ tịch Hội đồng Trị sự GHPGVN.
Năm 12 tuổi, Hòa thượng từ Paksé về Sài Gòn, sau đó trở lại Huế, tu học tại chùa Từ Đàm với Trưởng lão Hòa thượng Thích Thiện Siêu (1921-2001), rồi vào học tại Phật học viện Trung phần Hải Đức (Nha Trang), Quảng Hương Già Lam (Sài Gòn).
Hòa thượng tốt nghiệp Viện Cao đẳng Phật học Sài Gòn (1964) do Hòa thượng Thích Nhất Hạnh (1926-2022) sáng lập; sau đó, tốt nghiệp phân khoa Phật học của Viện Đại học Vạn Hạnh khi chỉ mới 22 tuổi.Năm 1970, với những công trình nghiên cứu, khảo luận có giá trị về Thiền học và Triết học Phật giáo, trong đó có tác phẩm đầu tay Đại cương về thiền quán và nổi bật hơn hết là Triết học về tánh Không, Hòa thượng được đặc cách bổ nhiệm Giáo sư thực thụ Viện Đại học Vạn Hạnh do Trưởng lão Hòa thượng Thích Minh Châu (1918-2012) làm Viện trưởng; là giáo sư trẻ nhất lúc bấy giờ.
Năm 1971, ngài được Hòa thượng Thích Minh Châu bổ nhiệm làm Tổng Thư ký tạp chí Tư Tưởng - cơ quan luận thuyết của Viện Đại học Vạn Hạnh. Bên cạnh đó, ngài cũng làm Thư ký tòa soạn, chủ bút, tham gia cộng tác với nhiều tờ báo, tạp chí nghiên cứu đương thời như: Vạn Hạnh, Hải Triều  m, Khởi Hành, Thời Tập,…
Hòa thượng Thích Tuệ Sỹ được biết nổi tiếng về sự uyên bác, thông thạo nhiều loại cổ ngữ lẫn sinh ngữ như: Hán văn, Phạn văn, Tạng văn, tiếng Anh, Pháp, Đức, Nga… Trong gần trọn cuộc đời, Hòa thượng dành phần lớn thời gian và tâm huyết của mình cho việc phiên dịch và chú giải kinh điển, đặc biệt là tạng kinh A-hàm. Các dịch phẩm nổi bật của Hòa thượng đã được xuất bản chính thức, đến với độc giả trong và ngoài nước…"""
    my_post_event_extraction = Post_Event_Extraction(post_4)
    # my_post_event_extraction._print()
    # print(my_post_event_extraction.post_event_extraction())
    print(my_post_event_extraction.handle_relevant_event({'Hòa thượng Thích Tuệ Sỹ đã viên tịch vào lúc 16:00 pm hôm 24/11/2023.': ['Hòa thượng Thích Tuệ Sỹ đã viên tịch vào lúc 16:00 pm hôm 24/11/2023.\nHòa thượng (HT) đã từng ở tù cộng sản hai lần, tổng cộng 17 năm tù. HT từng là giáo sư của đại học Vạn Hạnh - Sài Gòn. Năm 1984, HT bị nhà cầm quyền Việt Nam bắt giam, đến năm 1988 bị tuyên án tử hình với tội danh "Hoạt động nhằm lật đổ chính quyền nhân dân", cùng với HT Thích Trí Siêu (Lê Mạnh Thát). Sau đó, do sự vận động của các tổ chức nhân quyền, bản án được giảm xuống thành chung thân. \nNgày 1/9/1998, HT được trả tự do từ trại Ba Sao-Nam Hà. Trước khi thả, nhà cầm quyền yêu cầu ông ký vào lá đơn xin ân xá. HT trả lời: “Không ai có quyền xét xử tôi, không ai có quyền ân xá tôi!”. Công an nói không viết đơn thì không thả, HT không viết và tuyệt thực. Nhà cầm quyền đã phải phóng thích ngài sau mười ngày tuyệt thực. \nĐược trả tự do nhưng HT vẫn bị quản thúc cho đến bây giờ. HT đã dành cả đời để tu tập và đấu tranh cho sự từ bi và công bằng cho đất nước. \nĐạo là đời, đời là đạo. Tâm tu hành nhưng thân xác vẫn phải lăn trải vào đời để hành đạo, đây mới là con đường chân chính, tu mà không hành thì chưa phải là tu, hành mà không hiểu về giáo lý thì chỉ là tu mù. Hy vọng HT sẽ là tấm gương cho những vị tu sĩ, cư sĩ ngày nay hiểu được tu hành chân chính là thế nào, với một xã hội đầy nhiễu nhương, sự tu hành đầy chông chênh, gập ghềnh, dễ bị cám dỗ và lạc lối. \nKính tiễn Hòa Thượng về miền an lạc.\n    ']}))