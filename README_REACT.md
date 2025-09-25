# ai assistant - react + python

ultra-minimal ai assistant with react frontend and python backend.

## features

- **document q&a**: analyze pdf documents using natural language
- **data analysis**: analyze csv data with natural language queries
- **ultra-minimal ui**: jetbrains mono, black/white, small lowercase text
- **react frontend** + **fastapi backend** architecture

## requirements

- **python 3.8+** with pip
- **node.js 16+** with npm
- **ollama** running on localhost:11434

## quick start

### 1. start ollama (required)
```bash
ollama serve
```

### 2. start backend (terminal 1)
```bash
chmod +x start_backend.sh
./start_backend.sh
```

### 3. start frontend (terminal 2)
```bash
chmod +x start_frontend.sh
./start_frontend.sh
```

### 4. open browser
- **frontend**: http://localhost:3000
- **backend api**: http://localhost:8000

## manual setup

### backend setup
```bash
cd backend
python -m venv env
source env/bin/activate  # or `env\Scripts\activate` on windows
pip install -r requirements.txt
pip install streamlit pandas faiss-cpu langchain langchain-community langchain-nomic
python -m uvicorn main:main --host 0.0.0.0 --port 8000 --reload
```

### frontend setup
```bash
cd frontend
npm install
npm start
```

## api endpoints

- `GET /health` - health check
- `GET /database/status` - check database status
- `GET /data/status` - check data file status
- `POST /documents/upload` - upload pdf documents
- `POST /data/upload` - upload csv data
- `POST /qa/chat` - document q&a chat
- `POST /data/chat` - data analysis chat
- `POST /database/reset` - reset database

## architecture

```
frontend (react)     backend (fastapi)     ai models
     |                      |                   |
http://localhost:3000 -> http://localhost:8000 -> ollama:11434
```

## usage

1. **document mode**: upload pdf files, ask questions about content
2. **data mode**: upload csv file, ask analytical questions about data
3. **minimal ui**: everything centered, small text, black/white only
4. **buttons**: look like text, underline on hover
5. **font**: jetbrains mono throughout

## troubleshooting

- **ollama not running**: start with `ollama serve`
- **cors errors**: backend includes cors middleware for localhost:3000
- **file upload fails**: check file permissions and formats (pdf/csv only)
- **python imports fail**: ensure all dependencies installed in backend environment