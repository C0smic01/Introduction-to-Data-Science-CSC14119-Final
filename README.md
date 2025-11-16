# Football Player Data Scraper

Một ứng dụng Python để thu thập dữ liệu thống kê cầu thủ bóng đá từ **FBref.com** và giá trị chuyển nhượng từ **Transfermarkt.com**.

## 📋 Tính năng

- **Web Scraping**: Tự động thu thập dữ liệu từ FBref.com cho nhiều giải đấu châu Âu
- **Dữ liệu Toàn Diện**: Bao gồm 30+ chỉ số thống kê cho mỗi cầu thủ
  - Thông tin cá nhân: tên, quốc tịch, tuổi, chiều cao, vị trí, chân sút
  - Thống kê hiệu suất: bàn thắng, kiến tạo, xG, xAG
  - Thống kê phòng ngự: cắt bóng, chặn, chiến tranh không khí
  - Thống kê thủ môn: cứu thua, tỷ lệ cứu, bàn thua
  - Thống kê chuyền bóng: chính xác chuyền, chuyền tiến bộ
  - Giá trị chuyển nhượng từ Transfermarkt
- **Xuất Dữ Liệu**: Lưu dữ liệu dưới định dạng CSV hoặc JSON
- **ID Cầu Thủ Duy Nhất**: Tạo ID định danh duy nhất cho mỗi cầu thủ dựa trên tên, ngày sinh, quốc tịch

## 📊 Các Giải Đấu Được Hỗ Trợ

- Premier League (Anh)
- La Liga (Tây Ban Nha)
- Serie A (Ý)
- Bundesliga (Đức)
- Ligue 1 (Pháp)
- Eredivisie (Hà Lan)
- Primeira Liga (Bồ Đào Nha)

## 🛠️ Yêu Cầu Hệ Thống

- Python 3.8 trở lên
- Google Chrome (cho Selenium WebDriver)
- Windows, macOS hoặc Linux

## 📦 Cài Đặt

### 1. Clone Repository

```bash
git clone https://github.com/C0smic01/Introduction-to-Data-Science-CSC14119-Final.git
cd Introduction-to-Data-Science-CSC14119-Final
```

### 2. Tạo Virtual Environment (Tùy Chọn Nhưng Được Khuyến Nghị)

#### Trên Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### Trên macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Cài Đặt Thư Viện Phụ Thuộc

```bash
pip install -r requirements.txt
```

**Các thư viện được sử dụng:**
- `selenium` - Điều khiển trình duyệt Chrome
- `beautifulsoup4` - Phân tích HTML
- `webdriver-manager` - Tự động quản lý ChromeDriver
- `requests` - HTTP client
- `pandas` - Xử lý dữ liệu (nếu cần)
- `lxml` - Parser HTML nâng cao

## 🚀 Cách Sử Dụng

### Chạy Script Chính

Mở terminal/Command Prompt và chạy:

```bash
python main.py
```

Script sẽ:
1. Kết nối tới FBref.com
2. Thu thập danh sách tất cả các CLB từ các giải đấu
3. Lấy thông tin cầu thủ từ mỗi CLB
4. Trích xuất thống kê chi tiết từ trang cá nhân của cầu thủ
5. Lấy giá trị chuyển nhượng từ Transfermarkt
6. Lưu kết quả vào CSV và JSON

### Các Tệp Đầu Ra

Sau khi chạy, bạn sẽ nhận được:
- `players.csv` - Dữ liệu cầu thủ dạng bảng
- `players.json` - Dữ liệu cầu thủ dạng JSON

### Cấu Hình Tùy Chỉnh

Chỉnh sửa `config.py` để thay đổi:

```python
# Giải đấu cần thu thập
LEAGUE_CONFIG = {
    "La Liga": "https://fbref.com/en/comps/12/La-Liga-Stats",
    "Premier League": "https://fbref.com/en/comps/9/Premier-League-Stats",
    # ... thêm các giải đấu khác
}

# Độ trễ giữa các request (tính bằng giây)
DELAY_BETWEEN_REQUESTS = 2
DELAY_BETWEEN_PLAYERS = 1
```

## 💻 Các Tệp Chính

| File | Mô Tả |
|------|-------|
| `main.py` | Script chính để bắt đầu quá trình scraping |
| `fbref_scraper.py` | Class `FBrefCrawler` - Scraper chính từ FBref.com |
| `transfermarkt_scraper.py` | Hàm lấy giá trị chuyển nhượng từ Transfermarkt |
| `config.py` | Cấu hình, schema dữ liệu, headers HTTP |
| `requirements.txt` | Danh sách thư viện phụ thuộc |

## 📝 Ví Dụ Sử Dụng

### Ví Dụ 1: Thu thập một giải đấu

```python
from fbref_scraper import FBrefCrawler

with FBrefCrawler(headless=True) as crawler:
    players = crawler.scrape_league(
        "Premier League", 
        "https://fbref.com/en/comps/9/Premier-League-Stats"
    )
    print(f"Collected {len(players)} players")
```

### Ví Dụ 2: Thu thập nhiều giải đấu

```python
from config import LEAGUE_CONFIG
from fbref_scraper import FBrefCrawler

with FBrefCrawler(headless=True) as crawler:
    players = crawler.scrape_all_leagues(LEAGUE_CONFIG)
    crawler.save_to_csv("all_players.csv")
    crawler.save_to_json("all_players.json")
```

### Ví Dụ 3: Lấy dữ liệu của cầu thủ cụ thể

```python
from fbref_scraper import FBrefCrawler

with FBrefCrawler(headless=True) as crawler:
    player = crawler.scrape_player_full(
        "https://fbref.com/en/players/.../player-name",
        league_name="Premier League",
        club_name="Manchester City"
    )
    print(player)
```

## 🔧 Xử Lý Sự Cố

### Vấn đề: Chrome WebDriver không tìm thấy
**Giải pháp**: Cài đặt `webdriver-manager`:
```bash
pip install webdriver-manager
```

### Vấn đề: Bị chặn bởi website
**Giải pháp**:
- Tăng `DELAY_BETWEEN_REQUESTS` trong `config.py`
- Chạy với `headless=False` để thấy gì đang xảy ra
- Kiểm tra User-Agent trong `HEADERS`

### Vấn đề: Không thể lấy giá trị chuyển nhượng
**Giải pháp**:
- Kiểm tra kết nối Internet
- Transfermarkt có thể bị chặn - thử với delay lâu hơn
- Tên cầu thủ có thể không chính xác

### Vấn đề: Không tìm thấy bảng dữ liệu
**Giải pháp**:
- FBref thay đổi cấu trúc HTML - bảng có thể ẩn trong comment
- Kiểm tra hàm `find_table_in_comments()` đang hoạt động không

## ⚙️ Cấu Trúc Dữ Liệu

Mỗi cầu thủ bao gồm 43 trường dữ liệu:

```json
{
  "player_id": "lionel-messi-a3f8e2",
  "player_name": "Lionel Messi",
  "age": 36,
  "nationality": "Argentina",
  "height": 170,
  "foot": "Left",
  "position": "CF,RW",
  "current_club": "Paris Saint-Germain",
  "league": "Ligue 1",
  "market_value": "€25M",
  "appearances": 35,
  "minutes_played": 2769,
  "minutes_per_game": 79.1,
  "goals": 27,
  "assists": 13,
  "goals_per_90": 0.88,
  "assists_per_90": 0.42,
  ...
}
```

## 📌 Lưu Ý Quan Trọng

⚠️ **Tuân Thủ Điều Khoản Dịch Vụ**:
- Kiểm tra `robots.txt` trước khi scraping
- Sử dụng độ trễ hợp lý giữa các request
- Không spam các server
- Chỉ sử dụng dữ liệu cho mục đích học tập/nghiên cứu

⚠️ **Độ Ổn Định**:
- Website có thể thay đổi cấu trúc HTML
- Cần cập nhật scraper nếu HTML thay đổi
- Một số thông tin có thể không khả dụng cho tất cả cầu thủ

## 📄 Giấy Phép

MIT License - Xem file LICENSE để chi tiết

## 👤 Tác Giả

**DeratSonder** - [GitHub Profile](https://github.com/C0smic01)

## 📚 Tài Nguyên

- [FBref.com](https://fbref.com/) - Dữ liệu thống kê bóng đá
- [Transfermarkt.com](https://www.transfermarkt.com/) - Giá trị chuyển nhượng
- [Selenium Documentation](https://www.selenium.dev/)
- [BeautifulSoup Documentation](https://www.crummy.com/software/BeautifulSoup/)