import pandas as pd
import numpy as np


# --- 0️⃣ Đọc file ---

INPUT_FILE = "D:\\DataSciences\\Introduction-to-Data-Science-CSC14119-Final\\Data Exploration\\football_players_dataset.csv"

# Thay 'your_file.csv' bằng đường dẫn tới file dữ liệu của bạn
df = pd.read_csv(INPUT_FILE)

# issues_found = []

# # --- 1️⃣ Kiểm tra các cột không được âm ---
# non_negative_cols = [
#     'age', 'height', 'market_value', 'appearances', 'minutes_played', 'minutes_per_game',
#     'goals', 'assists', 'shots', 'shots_on_target', 'key_passes',
#     'tackles', 'interceptions', 'clearances', 'aerial_wins',
#     'clean_sheets', 'saves', 'goals_conceded', 'passes_completed', 'progressive_passes'
# ]

# print("\n* Kiểm tra giá trị âm:")
# for col in non_negative_cols:
#     if col in df.columns:
#         negative_count = (df[col] < 0).sum()
#         if negative_count > 0:
#             print(f"  ⚠️ '{col}': {negative_count} giá trị âm")
#             issues_found.append(f"{col}: {negative_count} giá trị âm")
#         else:
#             print(f"  '{col}': Không có giá trị âm")

# # --- 2️⃣ Kiểm tra các cột tỷ lệ % (0–100) ---
# percentage_cols = ['aerial_win_rate', 'pass_accuracy', 'save_percentage']

# print("\n* Kiểm tra tỷ lệ % (0–100):")
# for col in percentage_cols:
#     if col in df.columns:
#         invalid_count = ((df[col] < 0) | (df[col] > 100)).sum()
#         if invalid_count > 0:
#             print(f"  ⚠️ '{col}': {invalid_count} giá trị ngoài 0-100%")
#             issues_found.append(f"{col}: {invalid_count} giá trị % không hợp lệ")
#         else:
#             print(f"  '{col}': Tất cả giá trị hợp lệ")

# # --- 3️⃣ Kiểm tra logic nghiệp vụ ---
# print("\n* Kiểm tra logic nghiệp vụ:")

# # Shots on target không thể lớn hơn shots
# if 'shots' in df.columns and 'shots_on_target' in df.columns:
#     invalid = (df['shots_on_target'] > df['shots']).sum()
#     if invalid > 0:
#         print(f"  ⚠️ {invalid} trường hợp 'shots_on_target' > 'shots'")
#         issues_found.append(f"shots_on_target > shots: {invalid}")
#     else:
#         print(f"  'shots_on_target' luôn ≤ 'shots'")

# # Goals không thể lớn hơn shots
# if 'goals' in df.columns and 'shots' in df.columns:
#     invalid = (df['goals'] > df['shots']).sum()
#     if invalid > 0:
#         print(f"  ⚠️ {invalid} trường hợp 'goals' > 'shots'")
#         issues_found.append(f"goals > shots: {invalid}")
#     else:
#         print(f"  'goals' luôn ≤ 'shots'")

# # Assists không thể lớn hơn key passes
# if 'assists' in df.columns and 'key_passes' in df.columns:
#     invalid = (df['assists'] > df['key_passes']).sum()
#     if invalid > 0:
#         print(f"  ⚠️ {invalid} trường hợp 'assists' > 'key_passes'")
#         issues_found.append(f"assists > key_passes: {invalid}")
#     else:
#         print(f"  'assists' luôn ≤ 'key_passes'")

# # Minutes per game không quá 90
# if 'minutes_per_game' in df.columns:
#     invalid = (df['minutes_per_game'] > 90).sum()
#     if invalid > 0:
#         print(f"  ⚠️ {invalid} trường hợp 'minutes_per_game' > 90")
#         issues_found.append(f"minutes_per_game > 90: {invalid}")
#     else:
#         print(f"  'minutes_per_game' luôn ≤ 90")

# # Clean sheets không quá appearances
# if 'clean_sheets' in df.columns and 'appearances' in df.columns:
#     invalid = (df['clean_sheets'] > df['appearances']).sum()
#     if invalid > 0:
#         print(f"  ⚠️ {invalid} trường hợp 'clean_sheets' > 'appearances'")
#         issues_found.append(f"clean_sheets > appearances: {invalid}")
#     else:
#         print(f"  'clean_sheets' luôn ≤ 'appearances'")

# # Saves không thể lớn hơn shots_on_target
# if 'saves' in df.columns and 'shots_on_target' in df.columns:
#     invalid = (df['saves'] > df['shots_on_target']).sum()
#     if invalid > 0:
#         print(f"  ⚠️ {invalid} trường hợp 'saves' > 'shots_on_target'")
#         issues_found.append(f"saves > shots_on_target: {invalid}")
#     else:
#         print(f"  'saves' luôn ≤ 'shots_on_target'")

# # --- 4️⃣ Kiểm tra missing values ---
# missing_df = pd.DataFrame({
#     'Column': df.columns,
#     'Missing_Count': df.isnull().sum(),
#     'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2),
#     'Data_Type': df.dtypes
# })

# missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)

# if len(missing_df) > 0:
#     print("\n* Các cột có missing values:")
#     print(missing_df.to_string(index=False))
# else:
#     print("\n✅ Không có missing values")

# # --- 5️⃣ Tóm tắt tất cả vấn đề ---
# if issues_found:
#     print("\n\n⚠️ TỔNG HỢP CÁC VẤN ĐỀ BẤT HỢP LÝ TRONG DATASET:")
#     for issue in issues_found:
#         print(" -", issue)
# else:
#     print("\n✅ Dataset không phát hiện vấn đề bất hợp lý.")

num_shots_zero = (df["shots"] == 0).sum()

print("Số dòng có shots = 0:", num_shots_zero)
