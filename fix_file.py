import re

with open("/home/ignacio82/projects/book/strength.qmd", "r", encoding="utf-8") as f:
    content = f.read()

# Replace non-breaking spaces
content = content.replace("\u00a0", " ")

# Fix typos
content = content.replace("obervations", "observations")
content = content.replace("The code bellow", "The code below")
content = content.replace("strenght.html", "strength.html")

with open("/home/ignacio82/projects/book/strength.qmd", "w", encoding="utf-8") as f:
    f.write(content)
print("File updated successfully.")
