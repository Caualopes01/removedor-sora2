# ClearFrame — Watermark Remover MVP

Webapp para remoção automática de marcas d'água em vídeos usando IA (LaMa inpainting).

## Arquitetura

```
frontend/   → HTML/CSS/JS puro → deploy na Vercel
backend/    → FastAPI + OpenCV + LaMa → deploy no HuggingFace Spaces
```

---

## 🚀 Deploy do Backend (HuggingFace Spaces)

1. Crie uma conta em [huggingface.co](https://huggingface.co)
2. Crie um novo **Space**:
   - Nome: `watermark-remover`
   - SDK: **Docker**
   - Hardware: **CPU Basic** (grátis) ou **T4 GPU** (pago, mas muito mais rápido)
3. Faça upload dos arquivos da pasta `backend/`:
   - `app.py`
   - `requirements.txt`
   - `Dockerfile`
   - `README.md`
4. O Space vai buildar automaticamente
5. Anote a URL: `https://SEU-USUARIO-watermark-remover.hf.space`

---

## 🚀 Deploy do Frontend (Vercel)

### Opção 1: Vercel via GitHub
1. Suba a pasta `frontend/` para um repositório GitHub
2. Acesse [vercel.com](https://vercel.com) e importe o repositório
3. Antes do deploy, edite `index.html`:
   ```js
   const API_BASE_URL = 'https://SEU-USUARIO-watermark-remover.hf.space';
   ```
4. Deploy automático ✅

### Opção 2: Vercel CLI
```bash
npm i -g vercel
cd frontend
vercel deploy
```

---

## 🛠️ Rodar localmente

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

### Frontend
```bash
# Edite API_BASE_URL no index.html para http://localhost:8000
# Abra index.html no browser ou use live-server
npx live-server frontend/
```

---

## Como funciona

1. **Detecção automática**: OpenCV analisa variações locais de contraste e bordas nos primeiros frames, isolando regiões anômalas nos cantos do vídeo
2. **Máscara consolidada**: OR de múltiplos frames para garantir cobertura total da marca
3. **Inpainting LaMa**: Modelo preenche a região mascarada de forma coerente, frame a frame
4. **Fallback OpenCV**: Se LaMa falhar, usa `cv2.inpaint()` como backup

---

## Melhorias futuras

- [ ] Seleção manual da região (canvas drag)
- [ ] Suporte a múltiplos arquivos
- [ ] Preview do resultado lado a lado
- [ ] Otimização com ProPainter (melhor coerência temporal)
- [ ] Cache de modelos entre requests
- [ ] Fila com Redis + Celery para múltiplos usuários
