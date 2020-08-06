# auther 小强同学
from wordcloud import WordCloud
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
word_text = ""
with open("word_cut_data.txt","r",encoding= "utf-8") as f_data:
    word_text = f_data.read()
    cloud_mask = np.array(Image.open("bc_img/heart.png"))

wc = WordCloud(
    background_color="white",
    max_words= 500,
    font_path = "utils/wb.ttf",
    min_font_size=15,
    max_font_size= 50,
    width= 800,
    height= 700,
    mask= cloud_mask
)

wc.generate(word_text)
wc.to_file("wordcloud.png")
img = Image.open("wordcloud.png")

plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()

if __name__ == "__main__":
    pass

