# Sử dụng các thuật toán tìm kiếm AI để giải bài toán 8-Puzzle
# Triệu Phúc Hiếu - 23110217
# 1. Mục tiêu
  Dự án nhằm sử dụng các thuật toán tìm kiếm AI để giải bài toán 8-Puzzle với mục đích triển khai, đánh giá, so sánh mức độ hiệu quả, năng suất của từng thuật toán. Các nhóm thuật toán bao gồm:
  - Tìm kiếm không có thông tin (Uniformed Search): Gồm có các thuật toán BFS, DFS, UCS và IDS.
  - Tìm kiếm có thông tin (Informed Search): Gồm có các thuật toán GBFS, A*, IDA*.
  - Tìm kiếm cục bộ (Local Search): Gồm có các thuật toán Simulated Annealing, Beam Search, Genetic Algorithm và nhóm thuật toán Hill Climbing: Simple, Steepest, Stochastic
  - Tìm kiếm trong môi trường phức tạp (Searching in Complex Environments): Gồm có các thuật toán AND-OR Graph Search, Searching for a partially observation, Sensorless.
  - Bài toán thỏa mãn ràng buộc (Constraint Satisfaction Problems - CSPs): Gồm có các thuật toán Min-conflicts search, Forward-Checking, Backtracking.
  - Học tăng cường (Reinforcement Learning - RL): Gồm có thuật toán Q-Learning.
# 2. Nội dung
  Một bài toán tìm kiếm 8-Puzzle thường có các thành phần chính:
  - Trạng thái (State): Ma trận 3x3.
  - Trạng thái ban đầu (Initial State): Trạng thái của ma trận lúc ban đầu.
  - Trạng thái đích (Goal State): Trạng thái mà bài toán cần đạt được.
  - Hành động (Actions): Các hành động có thể có của trạng thái như L (Left), R (Right), U (Up), D (Down).
  - Kiểm tra đích (Goal Test): Kiểm tra xem trạng thái đang xét có phải trạng thái đích hay chưa.
  - Chi phí đường đi (Path Cost): Tổng chi phí của các bước đi từ trạng thái ban đầu đến trạng thái hiện tại.
  - Giải pháp (Solution): Một chuỗi các hành động (LRUD...) để giải bài toán từ trạng thái ban đầu đến trạng thái đích. 
# 2.1.  Tìm kiếm không có thông tin (Uniformed Search)
Minh họa:
![UNFS_2](https://github.com/user-attachments/assets/13c3a781-a9a5-457c-afcd-56030fd57107)
Biểu đồ so sánh khi chạy thuật toán:
  
![image](https://github.com/user-attachments/assets/e6d56b60-4958-4ca5-84a0-a3b63a9c4948)
  
**Nhận xét:**
+ Tìm kiếm theo chiều rộng (BFS - Breadth-First Search): Thời gian thực hiện và không gian trạng thái tương đối nhỏ bởi vì thuật toán này tìm kiếm theo lớp, đảm bảo tìm kiếm được lời giải tối ưu nhưng thường tốn bộ nhớ lớn.
+ Tìm kiếm theo chiều sâu (DFS - Depth-First Search): Thời gian thực hiện và không gian trạng thái lớn, vượt trội so với các thuật toán khác bởi vì DFS ưu tiên đi sâu theo nhánh nên dễ rơi vào vòng lặp hoặc đi sai hướng và khám phá các trạng thái không cần thiết.
+ Tìm kiếm theo chi phí đồng đều (UCS - Uniform Cost Search): UCS tương tự BFS nhưng có thêm chi phí đường đi, nếu chi phí giữa các trạng thái bằng nhau, UCS sẽ hoạt động tương tự như BFS.
+ Tìm kiếm lặp sâu dần (IDS - Iterative Deepening Search): IDS là sự kết hợp của BFS và DFS nhằm tiết kiệm bộ nhớ và tìm được lời giải tối ưu.
  
**Như vậy, các thuật toán BFS, UCS, IDS trong nhóm thuật toán này tối ưu hơn DFS**
# 2.2. Tìm kiếm có thông tin (Informed Search)
Minh họa: 
![ISS_1](https://github.com/user-attachments/assets/e5157cda-779a-4471-84a5-0121011a7381)
Biểu đồ so sánh khi chạy thuật toán:
  
![image](https://github.com/user-attachments/assets/d92d443a-ee11-4f64-8fdd-f418f4470aeb)
  
**Nhận xét**
+ Tìm kiếm tham lam theo chiều tốt nhất (GBFS - Greedy Best-First Search): Thời gian thực hiện nhỏ nhất nhưng không gian trạng thái lại lớn nhất trong nhóm thuật toán bởi vì GBFS chỉ dựa vào hàm heuristic (ước lượng đến đích) mà không xét chi phí đã đi qua, nên đôi khi không tối ưu và khám phá nhiều trạng thái sai hướng.
+ Tìm kiếm A* (A-Star): A* sử dụng kết hợp giữa chi phí đã đi và ước lượng còn lại (f(n) = g(n) + h(n)) nên thường tìm được lời giải tối ưu nếu heuristic tốt, cân bằng được giữa tốc độ và độ chính xác.
+ Tìm kiếm lặp sâu theo heuristic (IDA - Iterative Deepening A): IDA* kết hợp giữa A* và IDS, hoạt động theo ngưỡng (threshold) và tái sử dụng DFS có kiểm soát. Tuy thực hiện nhiều lần nhưng mỗi lần dùng ít bộ nhớ.

**Như vậy, các thuật toán trong nhóm thuật toán này đều có ưu điểm và nhược điểm, nếu như tìm kiếm thời gian nhanh thì không gian trạng thái lại nhiều (GBFS), tìm kiếm thời gian chậm thì không gian lại ít (IDA) và A-Star cân bằng tốt cả hai.**
# 2.3. Tìm kiếm cục bộ (Local Search)
Minh họa: 

![LCL_1111](https://github.com/user-attachments/assets/deb75fd5-d951-472e-8d73-c776397a70ba)
![LCL_2](https://github.com/user-attachments/assets/3e0578b1-f089-4ab1-b4d9-3cccf8c79d6f)
Biểu đồ so sánh khi chạy thuật toán:

![image](https://github.com/user-attachments/assets/fa27b310-46ce-4969-9e33-b9d3a6d8402d)

**Nhận xét**
+ Simple Hill Climbing (Simple Hill Climbing): Thuật toán đơn giản, chỉ đi theo hướng tăng mà không lùi, nên có thể nhanh chóng bị kẹt ở cực trị địa phương. Hiệu suất tốt nhưng khả năng tìm lời giải toàn cục thấp.
+ Steepest Ascent Hill Climbing (Steepest Hill Climbing): So với Simple Hill Climbing, Steepest Hill Climbing đánh giá tất cả các hàng xóm để chọn hướng tốt nhất, nhưng vẫn dễ bị kẹt ở cực trị. Cải tiến hơn Simple nhưng chưa đủ.
+ Stochastic Hill Climbing (Stochastic Hill Climbing): Chọn ngẫu nhiên các bước tăng tốt, nên có thể thoát khỏi một số cực trị địa phương so với Hill Climbing truyền thống, tối ưu hơn.
+ Simulated Annealing: Cho phép chấp nhận lời giải kém hơn với xác suất giảm dần, từ đó tránh được cực trị địa phương tốt hơn Hill Climbing. Chậm hơn nhưng chính xác hơn.
+ Beam Stochastic: Duy trì nhiều trạng thái đồng thời và chọn hướng đi tốt nhất, nên có khả năng khám phá không gian rộng hơn Hill Climbing. Cân bằng giữa hiệu suất và tối ưu.
+ Genetic Algorithm: Dựa vào quần thể, lai ghép, đột biến nên mất thời gian và bộ nhớ hơn nhưng tăng khả năng tìm lời giải toàn cục. Mặc dù chậm và tốn tài nguyên nhưng rất tối ưu.

**Như vậy, các thuật toán tìm kiếm cục bộ (Local Search) đều có ưu và nhược điểm riêng: những thuật toán như Hill Climbing có thời gian thực hiện rất nhanh nhưng dễ mắc kẹt tại cực trị địa phương, trong khi các thuật toán như Genetic Algorithm, Beam Stochastic, Simulated Annealing tốn nhiều thời gian và không gian hơn nhưng lại có khả năng tìm ra lời giải tối ưu cao hơn.**

  

  


