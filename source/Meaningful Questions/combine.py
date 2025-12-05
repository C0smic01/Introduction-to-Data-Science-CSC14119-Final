import nbformat

path1 = r"D:\DataSciences\Introduction-to-Data-Science-CSC14119-Final\source\Meaningful Questions\AskingQuestionImproved.ipynb"
path2 = r"D:\DataSciences\Introduction-to-Data-Science-CSC14119-Final\source\Meaningful Questions\‎Question4-5-6.ipynb"

with open(path1, "r", encoding="utf-8") as f1:
    nb1 = nbformat.read(f1, as_version=4)

with open(path2, "r", encoding="utf-8") as f2:
    nb2 = nbformat.read(f2, as_version=4)

nb1.cells.extend(nb2.cells)

output_path = r"D:\DataSciences\Introduction-to-Data-Science-CSC14119-Final\source\Meaningful Questions\merged.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbformat.write(nb1, f)

print("✅ Merge thành công!")
