# %% [markdown]
# # In-Context Learning
# 
# 
# In-context learning is a generalisation of few-shot learning where the LLM is provided a context as part of the prompt and asked to respond by utilising the information in the context.
# 
# * Example: *"Summarize this research article into one paragraph highlighting its strengths and weaknesses: [insert article text]”*
# * Example: *"Extract all the quotes from this text and organize them in alphabetical order: [insert text]”*
# 
# A very popular technique that you will learn in week 5 called Retrieval-Augmented Generation (RAG) is a form of in-context learning, where:
# * a search engine is used to retrieve some relevant information
# * that information is then provided to the LLM as context
# 

# %% [markdown]
# In this example we download some recent research papers from arXiv papers, extract the text from the PDF files and ask Gemini to summarize the articles as well as provide the main strengths and weaknesses of the papers. Finally we print the summaries to a local html file and as markdown.

# %%
!pip install requests bs4 google-generativeai pypdf

# %%
import os
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from urllib.request import urlopen, urlretrieve
from IPython.display import Markdown, display
from pypdf import PdfReader
from datetime import date
from tqdm import tqdm
# from google.colab import userdata

# %%
API_KEY = os.environ.get("GEMINI_API_KEY")
# API_KEY = userdata.get('GEMINI_API_KEY')
genai.configure(api_key=API_KEY)

# %% [markdown]
# We select those papers that have been featured in Hugging Face papers.

# %%
BASE_URL = "https://huggingface.co/papers"
page = requests.get(BASE_URL)
soup = BeautifulSoup(page.content, "html.parser")
h3s = soup.find_all("h3")

papers = []

for h3 in h3s:
    a = h3.find("a")
    title = a.text
    link = a["href"].replace('/papers', '')

    papers.append({"title": title, "url": f"https://arxiv.org/pdf{link}"})

# %% [markdown]
# Code to extract text from PDFs.

# %%
def extract_paper(url):
    html = urlopen(url).read()
    soup = BeautifulSoup(html, features="html.parser")

    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out

    # get text
    text = soup.get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return text


def extract_pdf(url):
    pdf = urlretrieve(url, "pdf_file.pdf")
    reader = PdfReader("pdf_file.pdf")
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def printmd(string):
    display(Markdown(string))

# %%
LLM = "gemini-2.5-flash"
model = genai.GenerativeModel(LLM)

# %% [markdown]
# We use Gemini to summarize the papers.

# %%
for paper in tqdm(papers):
    try:
        paper["summary"] = model.generate_content("Summarize this research article into one paragraph without formatting highlighting its strengths and weaknesses. " + extract_pdf(paper["url"])).text
    except:
        print("Generation failed")
        paper["summary"] = "Paper not available"

# %% [markdown]
# We print the results to a html file.

# %%
page = f"<html> <head> <h1>Daily Dose of AI Research</h1> <h4>{date.today()}</h4> <p><i>Summaries generated with: {LLM}</i>"
with open("papers.html", "w") as f:
    f.write(page)
for paper in papers:
    page = f'<h2><a href="{paper["url"]}">{paper["title"]}</a></h2> <p>{paper["summary"]}</p>'
    with open("papers.html", "a") as f:
        f.write(page)
end = "</head>  </html>"
with open("papers.html", "a") as f:
    f.write(end)

# %% [markdown]
# We can also print the results to this notebook as markdown.

# %%
for paper in papers:
    printmd("**[{}]({})**<br>{}<br><br>".format(paper["title"],
                                                paper["url"],
                                                paper["summary"]))


