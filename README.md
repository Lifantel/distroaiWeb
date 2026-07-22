> [!IMPORTANT]
> bu site bazen yanıltıcı cevaplar verebilir gerçek bir analiz istiyorsanız terminal üzerinden olan ve ana proje olan
> https://github.com/Lifantel/distroai reposunu ziyaret edin.


# distroai — statik sinir ağı çıkarımı

Bu site, PyTorch ile eğitilmiş bir MLP'yi hiçbir backend veya inference sunucusu olmadan, tamamen tarayıcıda çalıştırır. Sunucuya giden tek şey dört statik dosyadır (`index.html`, `style.css`, `app.js`, `model.js`); model çıkarımı kullanıcının cihazında yapılır.

## Mimari

`repo/src/distroai.py` içindeki ağ:

```
Linear(14 → 32) → ReLU → Dropout
Linear(32 → 16) → ReLU → Dropout
Linear(16 → 10)
→ Softmax
```

14 giriş, 14 soruya karşılık gelir (1-5 arası cevap, `{1:0.0, 2:0.25, 3:0.5, 4:0.75, 5:1.0}` ile normalize edilir). 10 çıkış, 10 distroya karşılık gelen olasılıklardır. Dropout katmanları yalnızca eğitimde aktif; eval modunda (ve dolayısıyla sitede) devre dışıdır.

## Eğitimden siteye

1. **Eğitim (bir kere, sunucu tarafında):** `dataset.csv` üzerinde sabit seed (42) ile 1000 epoch eğitildi. Validasyon doğruluğu %94.8.
2. **Ağırlık çıkarımı:** Eğitilmiş `nn.Linear` katmanlarının `weight` ve `bias` tensörleri `.tolist()` ile Python listesine, oradan JSON'a çevrildi.
3. **Gömme:** JSON, `model.js` içinde `const MODEL_WEIGHTS = [...]` olarak sabitlendi. Üç katman, her biri `{w: [[...]], b: [...]}` formatında (w: çıkış×giriş matrisi).
4. **Yeniden implementasyon:** `app.js` içinde `linear()`, `relu()`, `softmax()` fonksiyonları saf JavaScript ile, PyTorch'un yaptığı matris çarpımı + bias toplamını birebir tekrar eder. Kütüphane, WASM, ONNX Runtime yok — düz `for` döngüleri.

```
predict(x) = softmax( W3 · relu(W2 · relu(W1 · x + b1) + b2) + b3 )
```

Bu, PyTorch modelinin `model.eval()` sonrası `forward()` çağrısıyla matematiksel olarak birebir aynıdır; farklı sonuç vermesi için tek risk kaynağı kayan nokta hassasiyeti olur ki 32-bit aralığında ölçülemeyecek kadar küçüktür.

## Neden bu yöntem (ONNX/TF.js değil)

Model 3 katman, ~700 parametre — bu ölçekte bir runtime kütüphanesi yüklemek (ONNX Runtime Web ~ birkaç MB) gereksiz. `model.js` 25KB, hiç bağımlılık yok, sayfa açılışı anında hazır.

## Dosya yapısı

| Dosya | İş |
|---|---|
| `index.html` | Sayfa iskeleti, boot/intro/quiz/result ekranları |
| `style.css` | Görsel tasarım |
| `app.js` | Quiz akışı, cevap toplama, forward pass, sonuç render |
| `model.js` | Eğitilmiş ağırlıklar (statik veri, kod değil) |

## Modeli güncellemek istersen

`dataset.csv` veya ağ mimarisi değişirse `model.js` yeniden üretilmeli:

```python
layers = [m for m in model.network if isinstance(m, nn.Linear)]
weights = [{"w": l.weight.tolist(), "b": l.bias.tolist()} for l in layers]
# JSON'a yaz, "const MODEL_WEIGHTS = " ile başlat, model.js'e kaydet
```

`app.js`'deki `linear/relu/softmax` fonksiyonlarına dokunmaya gerek yok; sadece katman sayısı veya boyutları değişirse `predict()` içindeki katman zinciri güncellenir.

## Sınırlar

- Model dondurulmuştur, kullanıcı etkileşimi ağırlıkları değiştirmez (online öğrenme yok).
- `dataset.csv` her değiştiğinde yeniden eğitim ve yeniden export gerekir, bu otomatik değildir.
