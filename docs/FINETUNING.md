# Router'ı fine-tune etme (küçük local model)

Bu rehber, **küçük bir açık kaynak modeli "strict-JSON router" olacak şekilde
eğitir** — yani `static / chitchat / rag / fallback` kararını veren ve entity
çıkaran `classify()` adımını. Fine-tune için en değerli yer burası: düşük
gecikmeli bir ses backend'inde bu kısmın **hızlı, ucuz ve local** olması gerekir
ve görev yeterince dar olduğu için 1B–3B'lik bir model bunu rahatça öğrenir.

```
embedding router (bedava, modelsiz)  →  SENİN fine-tune ettiğin küçük LLM (router fallback)  →  daha büyük LLM (cevap üretir)
```

## 0. Ön koşullar
- Eğitim için **CUDA GPU'lu bir makine** (ücretsiz Colab T4 olur; ya da kiralık/kendi GPU'n).
- Modeli **çalıştıracak** (serve edecek) makinede **Ollama**. (1B bir modeli serve etmek için CPU yeterli.)
  Kurulum: https://ollama.com/download

## 1. Eğitim verisini hazırla
Veri script'i, asistan cevabı tam olarak modelin üretmeyi öğrenmesi gereken
`RouteDecision` JSON'u olan, chat formatında JSONL üretir:

```bash
# Dengeli train + sızıntısız (leak-free) eval seti üret:
python scripts/prepare_dataset.py --source mix --count 2000 --balance \
    --per-route 300 --eval-split 0.15 --output data/generated/router.jsonl
# -> data/generated/router_train.jsonl  (1200; her route'tan dengeli 300)
# -> data/generated/router_eval.jsonl   (held-out; train ile çakışmaz)
```

Pipeline: **dedupe → route bazında train/eval böl → SADECE train'i dengele**.
Böylece eval'deki hiçbir mesaj train'de yer almaz (sızıntı yok). `--source mix`,
gerçek Bitext dilini (rag/static) + şablonları (chitchat/fallback) birleştirir.

Her satır şöyle görünür:
```json
{"messages":[{"role":"system","content":"<router talimatları>"},
             {"role":"user","content":"Can I get a refund for order 61750?"},
             {"role":"assistant","content":"{\"route\":\"static\",\"intent\":\"refund_policy\",\"confidence\":0.83,\"entities\":[{\"type\":\"order_id\",\"value\":\"61750\"}]}"}],
 "route":"static","intent":"refund_policy","confidence":0.83,"entities":[...]}
```

**Gerçek kalite için veriyi güçlendir** (sentetik üretici sadece başlangıç noktası):
- `scripts/prepare_dataset.py` içine daha fazla şablon/intent ekle.
- Açık kaynak bir destek veri seti karıştır (ör. Hugging Face `bitext/Bitext-customer-support-llm-chatbot-training-dataset`) ve örneklerini **kendi 4 yoluna eşle**.
- En iyisi: birkaç yüz **gerçek** (anonimleştirilmiş) mesajı kendi yollarınla etiketle. Gerçek veri, sentetiği döver.
- Yolları (route) kabaca dengede tut ve değerlendirme için ~%10'unu ayır.

## 2. Eğit (GPU'lu makinede)

> 💡 **En kolay yol — Google Colab (ücretsiz GPU):** `notebooks/finetune_router_colab.ipynb`
> dosyasını [colab.research.google.com](https://colab.research.google.com) → *File → Upload
> notebook* ile aç, *Runtime → Change runtime type → T4 GPU* seç, hücreleri sırayla çalıştır;
> sorulduğunda `router_train.jsonl`'i yükle. Aşağıdaki CLI script'i aynı işi yapar:

```bash
pip install "unsloth[colab-new]" "trl<0.10" datasets
python scripts/finetune_router.py \
    --data data/generated/router_train.jsonl \
    --base-model unsloth/Qwen2.5-1.5B-Instruct \
    --epochs 3 \
    --export-gguf
```
Bu, temel modeli LoRA ile fine-tune eder ve `router-gguf/` altına quantize
edilmiş bir **GGUF** dosyası yazar. **Qwen2.5-1.5B**, JSON üretiminde güçlü ve
Apache-2.0 lisanslı (router için ideal). Maksimum hız için
`unsloth/Qwen2.5-0.5B-Instruct`, çok dilli (TR) için `google/gemma-3-1b-it`.

> **LoRA nedir?** Modelin tamamını değil, üzerine küçük bir "adaptör" katmanı
> eğitirsin. Çok daha ucuz ve hızlıdır; küçük bir GPU bile yeter.

## 3. (Önerilir) Değerlendir
> 💡 Colab notebook bunu **otomatik** yapar: eğitim ÖNCESİ (ham model) ve SONRASI
> (fine-tuned) `router_eval.jsonl` üzerinde macro route-accuracy ölçüp "+X puan
> iyileşme" basar.

`router_eval.jsonl` üzerinde, her kullanıcı mesajını modele ver ve tahmin ettiği
`route`'u gold etiketle karşılaştır. **route doğruluğu** ve **geçerli-JSON oranını**
ölç. Eval doğal dağılımda (rag ağırlıklı) olduğundan, tek bir genel doğruluk yerine
**route bazında doğruluğun ortalamasını** (macro-average) raporla — yoksa "hep rag
de" diyen bir model yanıltıcı şekilde yüksek görünür. Yalnızca her iki metrik de
temel modelden belirgin biçimde iyiyse yayına al.

## 4. Ollama ile çalıştır (serve)
GGUF'u gösteren bir `Modelfile` oluştur:

```
FROM ./router-gguf/unsloth.Q4_K_M.gguf
PARAMETER temperature 0
```

Sonra modeli kaydet:
```bash
ollama create support-router -f Modelfile
ollama run support-router "How do I reset my password?"   # hızlı kontrol
```

## 5. Projeye bağla
`.env` içinde:
```
APP_LLM_PROVIDER=local
APP_LOCAL_BASE_URL=http://localhost:11434/v1
APP_LOCAL_MODEL=llama3.2:1b              # chitchat/RAG cevabı üretmek için
APP_LOCAL_ROUTER_MODEL=support-router    # SENİN fine-tune ettiğin router
```
Uygulamayı başlat ve bir istek akıt:
```bash
uvicorn app.main:app --reload
curl -N -X POST localhost:8000/chat -H "Content-Type: application/json" \
     -d '{"message":"How do I reset my password?"}'
```
Artık ucuz embedding router emin olamadığında `metadata` olayındaki
`"source":"llm"` **senin fine-tune ettiğin** router'dan gelecek.

## Bu, mimariye nasıl oturuyor?
Kodun geri kalanı **hiç değişmez** — `LocalLLMProvider` (`app/services/llm.py`),
mock ve Claude sağlayıcılarıyla **aynı `LLMProvider` Protocol**'ünü uygular;
grafik, streaming ve API'ye dokunulmaz. "Kendi fine-tune ettiğim modeli takmayı"
bir yeniden-yazım değil de bir config değişikliği yapan şey, işte bu Protocol
"prizi"dir.
