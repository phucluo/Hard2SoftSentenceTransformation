import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# =========================
# 1. Dataset song song
# =========================
love_pairs = [
    {
        "id": 1,
        "hard": "Anh đang bận, đừng nhắn nữa.",
        "soft": "Anh đang hơi bận một chút, em cho anh xin ít thời gian, xong việc anh nhắn lại em liền nhé. 💌"
    },
    {
        "id": 2,
        "hard": "Em phiền quá.",
        "soft": "Chắc do anh hơi mệt nên phản ứng chưa được tốt, nhưng anh biết em quan tâm anh, cảm ơn em nhiều nha. 🫶"
    },
    {
        "id": 3,
        "hard": "Anh không muốn đi với em đâu.",
        "soft": "Hôm nay anh hơi đuối, mình hẹn dịp khác để anh có nhiều năng lượng dành cho em hơn nhé. 💖"
    },
    {
        "id": 4,
        "hard": "Đừng gọi cho anh nữa.",
        "soft": "Anh cần chút không gian để sắp xếp lại mọi thứ, xong anh sẽ chủ động liên lạc với em nhé."
    },
    {
        "id": 5,
        "hard": "Em nói nhiều quá.",
        "soft": "Tự nhiên hôm nay anh hơi mệt nên tiếp thu chậm, mình nói từ từ từng chuyện một được không em? 🥺"
    },
    {
        "id": 6,
        "hard": "Anh không quan tâm mấy chuyện đó.",
        "soft": "Anh chưa hiểu rõ lắm chuyện này, em kể thêm cho anh với để anh hiểu em hơn nha."
    },
    {
        "id": 7,
        "hard": "Thích thì làm, anh chịu.",
        "soft": "Anh hơi bối rối nên chưa biết quyết sao, hay là mình bàn thêm chút để tìm cách tốt nhất cho cả hai ha?"
    },
    {
        "id": 8,
        "hard": "Tuỳ em.",
        "soft": "Anh tin vào lựa chọn của em, nếu em cần ý kiến thêm thì anh luôn sẵn sàng góp ý nè. 😊"
    },
    {
        "id": 9,
        "hard": "Anh mệt em lắm rồi.",
        "soft": "Dạo này anh hơi áp lực nên dễ cáu, nhưng anh vẫn rất trân trọng em. Mình cùng tìm cách để cả hai đỡ mệt hơn nhé."
    },
    {
        "id": 10,
        "hard": "Đừng nhắn nữa, phiền.",
        "soft": "Anh đang bị overload chút, cho anh xin tạm nghỉ điện thoại, tối anh ổn hơn anh nhắn em nha."
    },
    {
        "id": 11,
        "hard": "Anh không muốn nghe em than nữa.",
        "soft": "Anh sợ mình không cho em được lời khuyên tốt nhất lúc này, nhưng anh vẫn luôn ở đây để lắng nghe em, mình nói chậm lại một chút nha."
    },
    {
        "id": 12,
        "hard": "Em ghen vô lý quá.",
        "soft": "Anh hiểu em quan tâm nên mới để ý nhiều vậy, mình cùng nói rõ cho nhau hiểu hơn để em yên tâm hơn nha."
    },
    {
        "id": 13,
        "hard": "Anh chán rồi.",
        "soft": "Anh đang thấy mối quan hệ mình có vài chỗ chưa ổn, mình thử ngồi lại nói chuyện để cải thiện được không em?"
    },
    {
        "id": 14,
        "hard": "Anh không thích kiểu em làm vậy.",
        "soft": "Có vài điều anh hơi chưa quen với cách em làm, mình bàn với nhau xem có cách nào hợp với cả hai hơn không em?"
    },
    {
        "id": 15,
        "hard": "Đừng làm phiền anh nữa.",
        "soft": "Anh đang cần tập trung một chút, em cho anh xin ít thời gian, xong anh quay lại với em nha. 💗"
    },
    {
        "id": 16,
        "hard": "Anh thấy em vô lý lắm.",
        "soft": "Anh hơi khó hiểu với cách em nhìn chuyện này, em giúp anh hiểu góc nhìn của em hơn được không?"
    },
    {
        "id": 17,
        "hard": "Anh không muốn nhắn tin nữa.",
        "soft": "Anh hơi mệt mắt vì điện thoại rồi, mình nói chuyện tiếp sau nhé, anh không muốn trả lời em trong trạng thái không tốt."
    },
    {
        "id": 18,
        "hard": "Anh không có thời gian cho em.",
        "soft": "Dạo này lịch của anh hơi dày, nhưng anh vẫn muốn sắp xếp thời gian cho em, mình cùng tìm khung giờ hợp lý hơn nha."
    },
    {
        "id": 19,
        "hard": "Em suy nghĩ nhiều quá.",
        "soft": "Anh hiểu em lo, nhưng nhiều khi em đang tự làm mình mệt hơn, để anh cùng em gỡ từng chuyện một nha."
    },
    {
        "id": 20,
        "hard": "Chuyện đó không quan trọng.",
        "soft": "Có vẻ anh chưa cảm nhận hết được tầm quan trọng của chuyện này với em, em kể kỹ hơn cho anh hiểu được không?"
    },
    {
        "id": 21,
        "hard": "Anh thấy em trẻ con quá.",
        "soft": "Có vài phản ứng của em làm anh hơi bất ngờ, mình thử tìm cách nói chuyện với nhau chín chắn hơn một xíu nha."
    },
    {
        "id": 22,
        "hard": "Anh không thích bạn em.",
        "soft": "Anh hơi lo khi em chơi với vài người bạn, chắc do anh sợ em bị ảnh hưởng, mình nói kỹ hơn để hiểu nhau hơn nhé."
    },
    {
        "id": 23,
        "hard": "Anh không muốn nói về chuyện đó nữa.",
        "soft": "Anh sợ nếu nói tiếp lúc này sẽ dễ làm em buồn, mình tạm dừng một chút, khi cả hai bình tĩnh hơn mình nói tiếp nha."
    },
    {
        "id": 24,
        "hard": "Kệ em.",
        "soft": "Anh tôn trọng quyết định của em, nếu em muốn nghe thêm góc nhìn của anh thì anh luôn sẵn sàng chia sẻ."
    },
    {
        "id": 25,
        "hard": "Em làm anh khó chịu.",
        "soft": "Có vài điều khiến anh hơi khó chịu, nhưng anh muốn nói với em một cách để mình cùng sửa, chứ không phải trách em."
    },
    {
        "id": 26,
        "hard": "Anh không muốn gặp em.",
        "soft": "Anh đang không ở trạng thái tốt nhất, anh sợ gặp em sẽ làm em buồn, cho anh xin thời gian ổn lại rồi mình gặp nhau nhé."
    },
    {
        "id": 27,
        "hard": "Đừng hỏi nữa.",
        "soft": "Anh hơi mệt khi nghĩ về chuyện này, mình tạm gác lại một chút, tối anh bình tĩnh hơn mình nói tiếp nha."
    },
    {
        "id": 28,
        "hard": "Anh lười nói chuyện với em.",
        "soft": "Hôm nay anh hơi cạn năng lượng, nên nói chuyện không được nhiệt tình như bình thường, em đừng nghĩ là anh không muốn nói với em nha."
    },
    {
        "id": 29,
        "hard": "Anh không thích em làm vậy với người khác.",
        "soft": "Anh hơi khó chịu khi thấy em như vậy với người khác, chắc do anh hơi nhạy cảm, mình cùng bàn xem đâu là ranh giới thoải mái cho cả hai nha."
    },
    {
        "id": 30,
        "hard": "Đừng đăng mấy cái đó nữa.",
        "soft": "Anh hơi lo khi thấy em đăng mấy bài đó, anh sợ người khác hiểu sai về em, mình nói với nhau xem có cách khác để em chia sẻ cảm xúc không nha."
    },
    {
        "id": 31,
        "hard": "Sao lúc nào em cũng than vậy?",
        "soft": "Anh nghe em than nhiều nên cũng lo cho em, hay mình cùng tìm cách để em đỡ áp lực hơn nha."
    },
    {
        "id": 32,
        "hard": "Anh thấy em yếu đuối quá.",
        "soft": "Anh thấy em đang rất nhạy cảm dạo này, anh muốn ở bên để em cảm thấy vững tin hơn."
    },
    {
        "id": 33,
        "hard": "Em làm vậy anh mất mặt.",
        "soft": "Lúc đó anh hơi ngại trước mọi người, lần sau mình trao đổi trước với nhau để cả hai cùng thoải mái hơn nha."
    },
    {
        "id": 34,
        "hard": "Anh không thích em than lên mạng.",
        "soft": "Anh hơi lo khi thấy em chia sẻ nhiều lên mạng, anh sợ người khác hiểu sai về em, mình thử tìm cách khác để em trút bớt mệt mỏi nha."
    },
    {
        "id": 35,
        "hard": "Đừng kể lể nữa, mệt lắm.",
        "soft": "Anh sợ nghe nhiều quá mà không giúp được gì sẽ làm em thất vọng, mình nói từng chuyện một để anh hiểu và chia sẻ với em tốt hơn nha."
    },
    {
        "id": 36,
        "hard": "Anh không muốn nói lại chuyện cũ.",
        "soft": "Chuyện đó làm anh khá buồn nên mỗi lần nhắc lại anh hơi khó chịu, mình thử tập trung vào cách giải quyết hiện tại được không em?"
    },
    {
        "id": 37,
        "hard": "Em drama quá.",
        "soft": "Anh biết em cảm xúc mạnh nên phản ứng vậy, mình cùng nhau học cách nói chuyện nhẹ nhàng hơn để cả hai đỡ mệt nhé."
    },
    {
        "id": 38,
        "hard": "Anh không chịu nổi tính em nữa.",
        "soft": "Có vài tính cách của em làm anh hơi khó thích nghi, nhưng nếu em đồng ý thì mình cùng nhau chỉnh sửa chút cho hợp nhau hơn nha."
    },
    {
        "id": 39,
        "hard": "Em hay làm quá mọi chuyện.",
        "soft": "Anh thấy đôi khi em nhìn mọi thứ nghiêm trọng hơn anh nghĩ, mình thử nhìn chuyện theo nhiều góc để bớt áp lực hơn nhé."
    },
    {
        "id": 40,
        "hard": "Anh không muốn giải thích nữa.",
        "soft": "Anh thấy mình đang hơi căng, nếu nói tiếp sợ sẽ không còn bình tĩnh. Mình tạm dừng một chút rồi nói lại khi cả hai dịu hơn nha."
    },
    {
        "id": 41,
        "hard": "Em làm anh mất tự do.",
        "soft": "Đôi lúc anh thấy mình hơi thiếu không gian riêng, mình cùng bàn cách để vừa gần gũi mà vẫn tôn trọng khoảng riêng của nhau nha."
    },
    {
        "id": 42,
        "hard": "Đừng kiểm soát anh nữa.",
        "soft": "Anh hiểu em lo cho anh, nhưng nếu mình tin nhau hơn một chút thì cả hai sẽ thấy thoải mái hơn đó em."
    },
    {
        "id": 43,
        "hard": "Em nhắn nhiều quá, anh ngợp.",
        "soft": "Tin nhắn của em nhiều làm anh thấy em rất quan tâm, nhưng đôi lúc anh hơi bị ngợp, mình điều chỉnh nhịp một chút cho hợp nhau nha."
    },
    {
        "id": 44,
        "hard": "Anh không muốn em can thiệp chuyện này.",
        "soft": "Chuyện này anh muốn tự mình xử lý trước, khi nào cần anh sẽ nhờ em giúp, anh không muốn em lo lắng thêm."
    },
    {
        "id": 45,
        "hard": "Em đừng xen vào chuyện bạn anh.",
        "soft": "Có vài chuyện liên quan tới bạn anh khá nhạy cảm, anh muốn giữ riêng một chút, nhưng anh vẫn luôn sẵn sàng kể cho em những gì anh có thể chia sẻ."
    },
    {
        "id": 46,
        "hard": "Anh không muốn em nói với gia đình anh.",
        "soft": "Một số chuyện anh vẫn chưa sẵn sàng chia sẻ với gia đình, mình từ từ tính để anh chuẩn bị tâm lý kỹ hơn nhé."
    },
    {
        "id": 47,
        "hard": "Em nghĩ linh tinh nhiều quá.",
        "soft": "Anh thấy em đang suy diễn theo hướng làm mình mệt hơn, để anh giải thích kỹ lại để em đỡ lo nha."
    },
    {
        "id": 48,
        "hard": "Anh bảo rồi, đừng hỏi nữa.",
        "soft": "Anh xin lỗi nếu câu trả lời trước chưa đủ rõ, nhưng hiện tại anh chưa có thêm thông tin gì hơn, để có gì mới anh sẽ nói ngay với em nha."
    },
    {
        "id": 49,
        "hard": "Anh không thích em nhắn kiểu đó.",
        "soft": "Có vài câu em nhắn làm anh hơi chạnh lòng, mình thử đổi cách diễn đạt một chút để cả hai đỡ tổn thương nhau hơn nhé."
    },
    {
        "id": 50,
        "hard": "Trễ vậy còn nhắn gì nữa.",
        "soft": "Giờ cũng khuya rồi, mình nghỉ ngơi chút để mai còn có năng lượng nói chuyện với nhau vui vẻ hơn nha."
    },
    {
        "id": 51,
        "hard": "Em không hiểu gì anh hết.",
        "soft": "Chắc anh chưa diễn đạt rõ nên em mới hiểu khác, để anh thử nói lại theo cách dễ hiểu hơn cho em nha."
    },
    {
        "id": 52,
        "hard": "Anh nói vậy mà em cũng không hiểu.",
        "soft": "Có thể cách anh giải thích chưa phù hợp với em, mình cùng tìm cách nói chuyện để cả hai bắt sóng nhau tốt hơn nhé."
    },
    {
        "id": 53,
        "hard": "Anh không thích em so sánh anh với người khác.",
        "soft": "Khi em so sánh anh với người khác, anh hơi chạnh lòng, mình tập trung vào chuyện của riêng hai đứa mình được không em?"
    },
    {
        "id": 54,
        "hard": "Em nhạy cảm quá.",
        "soft": "Anh thấy em rất tinh ý và dễ bị ảnh hưởng cảm xúc, anh sẽ cố gắng cẩn thận lời nói hơn để em đỡ buồn nha."
    },
    {
        "id": 55,
        "hard": "Anh không muốn em nhắc chuyện đó nữa.",
        "soft": "Mỗi lần nhắc đến chuyện đó anh lại thấy nặng lòng, mình tạm cất nó lại, khi anh sẵn sàng hơn mình nói với nhau thật kỹ nhé."
    },
    {
        "id": 56,
        "hard": "Em làm anh mất tập trung.",
        "soft": "Tin nhắn của em làm anh cứ muốn trả lời ngay, nên đôi lúc anh bị phân tâm. Mình canh thời gian phù hợp hơn để nói chuyện cho thoải mái nha."
    },
    {
        "id": 57,
        "hard": "Anh không muốn em nói với người khác chuyện của mình.",
        "soft": "Chuyện của hai đứa mình anh muốn giữ riêng một chút, nếu cần góp ý thì mình chọn người thật sự tin tưởng rồi cùng bàn em nhé."
    },
    {
        "id": 58,
        "hard": "Em lúc nào cũng nghi ngờ anh.",
        "soft": "Anh cảm giác mình chưa tạo được đủ sự yên tâm cho em, mình cùng nói rõ với nhau để xây lại niềm tin nha."
    },
    {
        "id": 59,
        "hard": "Anh chịu hết nổi rồi.",
        "soft": "Anh đang rất căng thẳng và dễ nổi nóng, mình tạm thời chậm lại một nhịp, rồi cùng nhau tìm cách giải quyết khi bình tĩnh hơn nha."
    },
    {
        "id": 60,
        "hard": "Anh không thích em nói chuyện kiểu đó.",
        "soft": "Một vài cách nói của em làm anh hơi buồn, mình cùng nhau điều chỉnh để nói năng với nhau nhẹ nhàng hơn nha."
    },
    {
        "id": 61,
        "hard": "Em làm anh thất vọng.",
        "soft": "Có vài chuyện xảy ra khác với kỳ vọng của anh, nhưng anh muốn hiểu lý do của em trước khi mình cùng tìm hướng sửa với nhau."
    },
    {
        "id": 62,
        "hard": "Anh không tin em nữa.",
        "soft": "Hiện tại niềm tin của anh đang bị lung lay, nhưng nếu em vẫn muốn, mình cùng từ từ xây dựng lại từng chút một nha."
    },
    {
        "id": 63,
        "hard": "Anh không muốn em nói chuyện với người đó.",
        "soft": "Anh hơi khó chịu và lo khi em thân với người đó, chắc do anh nhạy cảm. Mình bàn với nhau về ranh giới để cả hai đều yên tâm nha."
    },
    {
        "id": 64,
        "hard": "Em làm quá mọi chuyện lên rồi.",
        "soft": "Anh nghĩ mình đang nhìn chuyện này khác nhau, nên cảm xúc cũng bị đẩy lên. Mình thử lùi lại một chút, nói từng ý để hiểu nhau hơn nha."
    },
    {
        "id": 65,
        "hard": "Anh không muốn giải thích thêm.",
        "soft": "Anh cảm thấy mình đã nói khá nhiều rồi nhưng vẫn chưa được hiểu đúng, anh xin nghỉ một chút để sắp xếp lại cách nói, rồi mình nói tiếp sau em nha."
    },
    {
        "id": 66,
        "hard": "Em hay trách móc quá.",
        "soft": "Nhiều lúc nghe em trách anh cũng buồn, chắc em đang chịu nhiều tủi thân. Mình nói với nhau theo hướng góp ý để anh dễ tiếp thu hơn nha."
    },
    {
        "id": 67,
        "hard": "Anh chán nghe em than rồi.",
        "soft": "Anh sợ nghe em than hoài mà không giúp gì được sẽ làm em thất vọng, mình cùng nhau tìm giải pháp cụ thể cho từng chuyện nha."
    },
    {
        "id": 68,
        "hard": "Anh không muốn em khóc nữa.",
        "soft": "Thấy em khóc anh xót lắm, mình cố gắng nói chuyện với nhau chậm hơn, nếu cần anh sẽ im lặng một lúc để em bớt xúc động rồi mình tiếp tục nha."
    },
    {
        "id": 69,
        "hard": "Em đừng gọi anh liên tục nữa.",
        "soft": "Khi em gọi nhiều quá anh hơi cuống và khó xử, mình hẹn nhau giờ nói chuyện cụ thể để anh toàn tâm với em hơn nha."
    },
    {
        "id": 70,
        "hard": "Anh không chịu nổi cách em ghen.",
        "soft": "Cách em ghen làm anh hơi ngợp, nhưng anh hiểu là vì em quan tâm. Mình cùng nhau thống nhất ranh giới và cách nói để cả hai dễ chịu hơn nha."
    },
    {
        "id": 71,
        "hard": "Em làm anh mệt mỏi.",
        "soft": "Thời gian này anh hơi kiệt sức nên dễ cáu hơn bình thường, không phải do em đâu. Mình cùng điều chỉnh lại nhịp sống chung cho nhẹ nhàng hơn nha."
    },
    {
        "id": 72,
        "hard": "Anh không muốn em đi chơi với nhóm đó.",
        "soft": "Anh hơi lo cho em khi đi với nhóm đó vì vài chuyện anh nghe được, mình cùng nhau nói kỹ hơn để em cân nhắc nha."
    },
    {
        "id": 73,
        "hard": "Em để ý chi tiết vụn vặt quá.",
        "soft": "Anh thấy em rất tinh ý, nhưng đôi khi mình bỏ qua vài chuyện nhỏ sẽ đỡ mệt hơn, mình chọn lọc những điều thật sự quan trọng để nói với nhau nha."
    },
    {
        "id": 74,
        "hard": "Anh không thích em check điện thoại của anh.",
        "soft": "Anh hiểu em muốn yên tâm, nhưng việc kiểm tra điện thoại làm anh thấy hơi bị xâm phạm. Mình thử xây thêm niềm tin bằng cách khác được không em?"
    },
    {
        "id": 75,
        "hard": "Anh không muốn em giữ thái độ đó với gia đình anh.",
        "soft": "Anh biết có vài điều em khó chịu, nhưng anh rất trân trọng gia đình. Mình cùng nhau tìm cách cư xử để vừa tôn trọng họ vừa không làm em tổn thương nha."
    },
    {
        "id": 76,
        "hard": "Em lúc nào cũng nghĩ tiêu cực.",
        "soft": "Anh thấy em hay nhìn về phía xấu trước nên dễ buồn, nếu em muốn, anh sẽ cố gắng cùng em nhìn thêm các khả năng tích cực hơn nữa nha."
    },
    {
        "id": 77,
        "hard": "Anh không muốn em nhắc người yêu cũ nữa.",
        "soft": "Mỗi lần nhắc đến người cũ anh hơi khó chịu, chắc do anh còn nhạy cảm. Mình tập trung vào hiện tại của hai đứa mình nha."
    },
    {
        "id": 78,
        "hard": "Em hay để bụng quá.",
        "soft": "Anh thấy em nhớ rất lâu những chuyện làm em buồn, anh sẽ cố gắng cẩn thận hơn, và mình cũng thử học cách buông nhẹ vài chuyện nhỏ để lòng nhẹ hơn nha."
    },
    {
        "id": 79,
        "hard": "Anh không muốn em trách móc trước mặt người khác.",
        "soft": "Khi em góp ý anh trước mặt mọi người anh hơi ngại, mình để dành những điều đó nói riêng với nhau, anh sẽ lắng nghe nghiêm túc hơn nha."
    },
    {
        "id": 80,
        "hard": "Em đừng bắt anh trả lời ngay lập tức.",
        "soft": "Đôi lúc anh cần thêm chút thời gian suy nghĩ trước khi trả lời, nếu được em cho anh chậm lại một nhịp để anh nói với em kỹ hơn nha."
    },
    {
        "id": 81,
        "hard": "Anh không muốn em lục đồ của anh.",
        "soft": "Anh hơi nhạy cảm với việc người khác động vào đồ cá nhân, mình bàn với nhau chỗ nào anh thoải mái để em giúp, chỗ nào anh muốn tự giữ nhé."
    },
    {
        "id": 82,
        "hard": "Em làm anh thấy bị áp lực.",
        "soft": "Anh biết em kỳ vọng ở anh nhiều nên anh tự tạo áp lực cho mình, mình cùng điều chỉnh lại kỳ vọng để cả hai không quá nặng nề nha."
    },
    {
        "id": 83,
        "hard": "Anh không muốn em so đo với người yêu người khác.",
        "soft": "Mỗi cặp đều có cách yêu riêng, anh muốn tập trung vào cách hai tụi mình xây dựng, thay vì so với ai khác, để em đỡ chạnh lòng hơn nhé."
    },
    {
        "id": 84,
        "hard": "Em đừng bắt anh phải giống người ta.",
        "soft": "Anh sẽ cố gắng tốt hơn mỗi ngày, nhưng vẫn là chính mình, hy vọng em đồng hành với anh trong hành trình đó, chứ không cần anh giống bất kỳ ai."
    },
    {
        "id": 85,
        "hard": "Anh không thích em nói chuyện kiểu mệnh lệnh.",
        "soft": "Khi em nói kiểu ra lệnh anh hơi bị khựng lại, nếu em nói theo kiểu nhờ vả nhẹ nhàng hơn, anh sẽ thấy vui và sẵn lòng hơn nhiều."
    },
    {
        "id": 86,
        "hard": "Em làm anh thấy không được tôn trọng.",
        "soft": "Có vài câu nói của em làm anh thấy mình chưa được trân trọng lắm, mình cùng chỉnh lại cách nói với nhau để cả hai đều cảm thấy được coi trọng nha."
    },
    {
        "id": 87,
        "hard": "Anh không muốn em kể bí mật của anh với ai.",
        "soft": "Những điều anh chia sẻ với em anh xem là rất riêng tư, anh mong điều đó chỉ ở lại giữa hai đứa mình thôi, như vậy anh sẽ yên tâm hơn khi mở lòng."
    },
    {
        "id": 88,
        "hard": "Em làm anh thấy tội lỗi hoài.",
        "soft": "Mỗi lần nhắc lại lỗi cũ anh càng cảm thấy có lỗi với em, anh muốn bù đắp bằng hành động hiện tại hơn là cứ bị nhắc lại mãi, mình thử hướng đến tương lai nha."
    },
    {
        "id": 89,
        "hard": "Anh không chịu nổi cách em nói khi nóng giận.",
        "soft": "Khi em giận, lời nói của em đôi khi làm anh buồn nhiều, mình thử thống nhất với nhau là lúc nào quá nóng thì tạm im lặng, đợi dịu lại rồi nói tiếp nha."
    },
    {
        "id": 90,
        "hard": "Em đừng ép anh phải luôn trả lời đúng ý em.",
        "soft": "Anh muốn chia sẻ suy nghĩ thật của mình, dù đôi khi khác với ý em. Mình cùng lắng nghe nhau nhiều hơn thay vì chỉ tìm câu trả lời đúng ý nha."
    },
    {
        "id": 91,
        "hard": "Anh không muốn em quyết định thay anh.",
        "soft": "Anh rất trân trọng ý kiến của em, nhưng có vài thứ anh muốn tự quyết để cảm thấy có trách nhiệm hơn, mình bàn và cùng thống nhất nha."
    },
    {
        "id": 92,
        "hard": "Em lúc nào cũng muốn kiểm soát anh.",
        "soft": "Anh hiểu em cần cảm giác an toàn, nhưng nếu mình tin nhau hơn thì anh sẽ thấy dễ thở hơn, và anh cũng sẽ chủ động chia sẻ với em nhiều hơn."
    },
    {
        "id": 93,
        "hard": "Anh không thích bị tra hỏi như vậy.",
        "soft": "Khi câu hỏi dồn dập quá anh thấy giống như đang bị tra khảo, mình đổi thành chia sẻ nhẹ nhàng hai chiều, anh sẽ kể cho em kỹ hơn nha."
    },
    {
        "id": 94,
        "hard": "Em đừng trách anh không lãng mạn.",
        "soft": "Anh biết mình không giỏi thể hiện như trên phim, nhưng anh luôn cố gắng quan tâm em theo cách của anh, nếu em gợi ý thêm thì anh càng dễ làm em vui hơn."
    },
    {
        "id": 95,
        "hard": "Anh không muốn cãi nhau nữa.",
        "soft": "Anh không muốn hai đứa cứ ở trong trạng thái đối đầu, anh muốn mình cùng đứng chung một phía để giải quyết vấn đề nhẹ nhàng hơn nha."
    },
    {
        "id": 96,
        "hard": "Em đừng suy diễn thêm nữa.",
        "soft": "Anh nghĩ càng suy diễn mình càng mệt, để anh kể em nghe mọi thứ theo những gì anh biết để em bớt phải tưởng tượng thêm nha."
    },
    {
        "id": 97,
        "hard": "Anh không muốn em phải kiểm tra anh mỗi ngày.",
        "soft": "Anh hiểu em muốn chắc chắn về anh, nhưng nếu ngày nào cũng kiểm tra anh sợ em càng mệt. Mình xây cách tin nhau bền vững hơn nha."
    },
    {
        "id": 98,
        "hard": "Em đừng đòi hỏi anh nhiều như vậy.",
        "soft": "Có những điều anh đang cố gắng hết sức nhưng vẫn chưa làm được như mong muốn của em, mình cùng đặt mục tiêu vừa phải hơn để anh phấn đấu từ từ nha."
    },
    {
        "id": 99,
        "hard": "Anh không muốn em tự so sánh mình với người khác.",
        "soft": "Đối với anh, em đã là rất đặc biệt rồi, anh không muốn em tự làm mình buồn vì so với ai khác. Mình tập trung vào việc em thấy vui và thoải mái là được nha."
    },
    {
        "id": 100,
        "hard": "Em làm anh thấy ngợp với cảm xúc của em.",
        "soft": "Cảm xúc của em rất mạnh nên đôi lúc anh hơi không theo kịp, nhưng anh muốn học cách hiểu em hơn, mình đi chậm từng bước để cả hai đều dễ chịu nha."
    },
]

love_df = pd.DataFrame(love_pairs)

# =========================
# 2. Load model + precompute embeddings
# =========================
@st.cache_resource
def load_embed_model():
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model = SentenceTransformer(model_name)
    return model

@st.cache_resource
def build_index(df):
    model = load_embed_model()
    hard_sentences = df["hard"].tolist()
    hard_embs = model.encode(hard_sentences, convert_to_tensor=True, show_progress_bar=False)
    return hard_sentences, df["soft"].tolist(), hard_embs

def rewrite_loving(user_text: str, top_k: int = 3, min_sim: float = 0.35):
    if not user_text or not user_text.strip():
        return []

    model = load_embed_model()
    hard_sents, soft_sents, hard_embs = build_index(love_df)

    query_emb = model.encode(user_text, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_emb, hard_embs)[0]

    top_k = min(top_k, len(hard_sents))
    top_results = torch.topk(cos_scores, k=top_k)

    results = []
    for score, idx in zip(top_results.values, top_results.indices):
        score_val = float(score.cpu().item())
        if score_val < min_sim:
            continue
        idx = int(idx)
        results.append(
            {
                "hard_template": hard_sents[idx],
                "soft_suggestion": soft_sents[idx],
                "similarity": score_val,
            }
        )
    return results

# =========================
# 3. Streamlit UI
# =========================
st.set_page_config(page_title="Cỗ máy dịch lời khó nói", page_icon="💌")

st.title("💌 Cỗ Máy Dịch Lời Khó Nói Sang Lời Dễ Thương")
st.write(
    """
Nhập một câu hơi **thẳng, khó nói** với người yêu, 
app sẽ gợi ý phiên bản **dịu dàng, dễ thương** hơn mà vẫn giữ ý chính.
"""
)

with st.expander("Xem một vài ví dụ có sẵn trong dataset"):
    st.table(love_df[["hard", "soft"]].head(5).rename(columns={"hard": "Câu khó nói", "soft": "Câu dễ thương"}))

user_input = st.text_area(
    "Nhập câu bạn định nhắn (càng thật càng tốt 😅)",
    height=100,
    placeholder="Ví dụ: Anh đang bận, đừng làm phiền anh nữa."
)

col1, col2 = st.columns([1, 1])
with col1:
    top_k = st.slider("Số gợi ý muốn xem", min_value=1, max_value=5, value=3)
with col2:
    min_sim = st.slider("Ngưỡng độ giống (similarity)", min_value=0.1, max_value=0.9, value=0.35, step=0.05)

if st.button("Biến hoá cho mềm hơn ✨"):
    if not user_input.strip():
        st.warning("Nhập gì đó trước đã nha. 🥹")
    else:
        with st.spinner("Đang suy nghĩ câu trả lời dễ thương hơn..."):
            candidates = rewrite_loving(user_input, top_k=top_k, min_sim=min_sim)

        if not candidates:
            st.info("Tớ chưa tìm được câu nào đủ giống trong dataset. Bạn có thể mở rộng dataset thêm sau này.")
        else:
            best = candidates[0]
            st.subheader("💘 Gợi ý phù hợp nhất")
            st.write(best["soft_suggestion"])

            st.caption(f"(Dựa trên template: “{best['hard_template']}”, similarity ≈ {best['similarity']:.2f})")

            if len(candidates) > 1:
                st.markdown("---")
                st.subheader("Các gợi ý khác")
                for i, cand in enumerate(candidates[1:], start=2):
                    with st.container(border=True):
                        st.markdown(f"**Phương án {i}**")
                        st.write(cand["soft_suggestion"])
                        st.caption(
                            f"Template: “{cand['hard_template']}”  ·  similarity ≈ {cand['similarity']:.2f}"
                        )

st.markdown("---")
st.markdown(
    """
This one for mah 4ever love, Quinvonnine <3
    """
)
