# **📚  Review Guide: AI Research Paper Evaluator**
An **AI-powered application** that analyzes research papers and provides **structured evaluations** using **Google Gemini AI**.  

✅ **Backend:** FastAPI (Python)  
✅ **Frontend:** React (Material UI)  
✅ **AI Model:** Google Gemini API  
✅ **Deployment:** Render (Backend) + Vercel (Frontend)  

---

## **📌 Features**
✔ **Upload Research Papers (PDFs)**  
✔ **AI Evaluation & Scoring** (Originality, Clarity, Impact, etc.)  
✔ **Chat-style UI for a modern feel**  
✔ **Fully responsive & user-friendly**  
✔ **Deployed using Render & Vercel**  

---

# **🚀 Backend (FastAPI)**
### **1️⃣ Installation & Setup**
#### **🔹 Clone the Repository**
```bash
git clone https://github.com/vidyavenkappa/fastapi-research-evaluator.git
cd fastapi-research-evaluator
```

#### **🔹 Create a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

#### **🔹 Install Dependencies**
```bash
pip install -r requirements.txt
```

#### **🔹 Create a `.env` File**
Inside the project directory, create a **`.env`** file:
```
GEMINI_API_KEY=your-google-gemini-api-key
```

#### **🔹 Run the FastAPI Server**
```bash
uvicorn main:app --reload
```
📌 **Backend will be available at:**  
```
http://127.0.0.1:8000
```

#### **🔹 Test API in Swagger UI**
Visit:
```
http://127.0.0.1:8000/docs
```

---

### **2️⃣ API Endpoints**
| Method | Endpoint       | Description |
|--------|---------------|-------------|
| POST   | `/upload/`    | Uploads a research paper (PDF) and evaluates it using Gemini AI |
| GET    | `/docs`       | Opens Swagger UI |

---

### **3️⃣ Deploy Backend on Render**
1. **Push your code to GitHub**
```bash
git add .
git commit -m "Initial commit"
git push origin main
```
2. **Go to [Render](https://render.com/)** → **New Web Service**  
3. **Connect GitHub repo** → Select **`fastapi-research-evaluator`**  
4. **Set Build Command:**
   ```bash
   pip install -r requirements.txt
   ```
5. **Set Start Command:**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```
6. **Add Environment Variable:**
   - `GEMINI_API_KEY` → Your Google Gemini API Key  
7. **Click Deploy!** 🎉  

📌 **Backend will be live at:**  
```
https://fastapi-research-evaluator.onrender.com
```

---

# **🎨 Frontend (React)**
### **1️⃣ Installation & Setup**
#### **🔹 Clone the Frontend Repo**
```bash
git clone https://github.com/vidyavenkappa/review-guide-frontend.git
cd react-research-evaluator
```

#### **🔹 Install Dependencies**
```bash
npm install
```

#### **🔹 Update API URL in `App.js`**
Modify `API_BASE_URL` in `App.js` to use the **live backend URL**:
```javascript
const API_BASE_URL = "https://fastapi-research-evaluator.onrender.com"; // Replace with Render API URL
```

#### **🔹 Run React App Locally**
```bash
npm start
```
📌 **Frontend will be available at:**  
```
http://localhost:3000
```

---

### **2️⃣ Deploy Frontend on Vercel**
1. Install **Vercel CLI**:
   ```bash
   npm install -g vercel
   ```
2. Login to **Vercel**:
   ```bash
   vercel login
   ```
3. Deploy React:
   ```bash
   vercel
   ```
4. **Get your frontend URL (e.g., `https://review-guide-frontend.vercel.app`)**

📌 **Now your React frontend is live! 🎯**

---

# **🎯 Final Steps**
- **Visit Frontend URL** (`https://review-guide-frontend.vercel.app`)
- **Upload a research paper** and verify AI evaluation works!
- **Test API in Swagger UI** (`https://fastapi-research-evaluator.onrender.com/docs`)

---

# **🛠 Tech Stack**
🔹 **Frontend:** React, Material UI  
🔹 **Backend:** FastAPI (Python), Google Gemini AI  
🔹 **Database:** (Optional) Can integrate PostgreSQL, Firebase, or MongoDB  
🔹 **Hosting:** Render (Backend) + Vercel (Frontend)  

---

# **🤝 Contributing**
Want to improve this project?  
1. **Fork the repo**  
2. **Create a new branch** (`feature-branch`)  
3. **Commit changes** (`git commit -m "Added new feature"`)  
4. **Push changes** (`git push origin feature-branch`)  
5. **Create a Pull Request!** 🎉  

---

# **📄 License**
This project is **open-source** under the **MIT License**.

---

## 🎉 **Now Your AI Research Evaluator is Fully Deployed! 🚀**
Let me know if you need **additional features or improvements!** 😊
