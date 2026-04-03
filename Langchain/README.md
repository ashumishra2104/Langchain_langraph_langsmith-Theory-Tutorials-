# 🦜 LangChain Basics

> **Course:** AI Product Management Cohort  
> **Instructor:** [Ashu Mishra](https://www.linkedin.com/in/ashumish/)  
> **Level:** Beginner — no prior Python experience needed

---

## 📌 What This Session Covers

This is a 2-hour hands-on session introducing LangChain from scratch. The focus is on understanding **what** each concept does — not memorising code.

| # | Concept | Real-life Analogy |
|---|---------|-------------------|
| 1 | **LLM** | Smart assistant locked in a room — you pass a note, they reply |
| 2 | **Temperature** | Creativity dial — from robot-precise to wildly imaginative |
| 3 | **Prompt Template** | WhatsApp birthday message — one template, fill in the name each time |
| 4 | **Chain** | Zomato delivery pipeline — each step feeds the next automatically |
| 5 | **Output Parser** | Meeting availability table — converts messy replies into clean structured data |

---

## 🗂️ Files in This Folder

| File | Description |
|------|-------------|
| `langchain_basics.ipynb` | Main Google Colab notebook — concepts + live demo + exercise |
| `LangChain_Basics.pptx` | Lecture slides (upload manually) |

---

## 🚀 How to Use the Notebook

1. Open `langchain_basics.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Run the install cell first and wait for it to finish
3. Replace `YOUR_OPENAI_API_KEY` with the key shared during the session
4. Run each cell one by one — read the markdown explanation before running the code
5. Complete the exercise at the end

---

## 🧠 The Magic Formula

```python
chain = prompt_template | llm | output_parser
result = chain.invoke({"your_variable": "your_value"})
```

Everything in LangChain is built around this one pattern.

---

## 📋 Case Study: Output Parser

One of the key demos in this session is a case study showing the difference between using and not using an Output Parser.

**Without parser:**
```
Type of result: <class 'langchain_core.messages.ai.AIMessage'>
Output: Here are five popular AI tools commonly used in product management...
```

**With parser:**
```
Type of result: <class 'list'>
AI Tools:
  1. Jira
  2. Asana
  3. Notion
  4. Airtable
  5. Productboard
```

Same question. Same AI. One parser added. Completely different result.

---

## ✏️ Exercise

Students build a chain that:
- Takes a **job role** as input
- Returns **top 5 skills** for that role as a clean list
- Uses: `ChatPromptTemplate` + `CommaSeparatedListOutputParser` + pipe chain

Only 3 things need to be changed from the template code — the system message, the human message, and the job role variable.

**Bonus:** Run it for 3 different roles and compare results.

---

## 🔮 What's Coming Next

| Session | Topic |
|---------|-------|
| Next | 🧠 Memory — make AI remember previous conversations |
| After | 📄 RAG — make AI answer from your own documents |
| Later | 🤖 Agents — give AI tools to search, calculate, and act |

---

## 🔗 Connect

- LinkedIn: [linkedin.com/in/ashumish](https://www.linkedin.com/in/ashumish/)
- GitHub: [github.com/ashumishra2104](https://github.com/ashumishra2104)
