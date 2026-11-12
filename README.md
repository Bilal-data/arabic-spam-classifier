# Arabic Spam / Ham Text Classifier (Learning Project)
# مصنّف رسائل عربية: سبام / عادية (مشروع تعليمي)

---

## English

### Overview
A small end-to-end NLP project that classifies short Arabic text messages as **spam** or **ham** (normal).
Built as part of my machine-learning learning path, using Python in Google Colab.

**Notebook:** [arabic_spam_classifier.ipynb](arabic_spam_classifier.ipynb) (the 13,240-tweet experiments, with outputs)

### Pipeline
1. **Data:** 50 hand-written Arabic messages (25 spam, 25 ham, balanced), including some Egyptian-dialect examples. Later tested on a public dataset of 13,240 Arabic tweets.
2. **Preprocessing:** remove Arabic diacritics (tashkeel), normalize alef variants, tokenize with NLTK.
3. **Features:** TF-IDF (`TfidfVectorizer`).
4. **Model:** Logistic Regression (`scikit-learn`).
5. **Saving:** model and vectorizer saved together with `joblib`.

### Results (honest numbers)
| Evaluation | Result | Notes |
|---|---|---|
| Single 70/30 split (9 test messages) | 100% | Too few test messages to trust. |
| 10 new "harder" messages (dialect, new phrasing) | 70% | Errors were normal messages labeled spam. |
| 5-fold cross-validation (44 messages) | mean 86.1% | Large spread, small data. |
| 5x5 repeated stratified CV, balanced data (50 messages) | **mean 87.2%** (min 70%, max 100%) | Wide spread. Gain over 86.1% is within noise. |
| **Larger real dataset:** 13,240 Arabic tweets (11,299 ham / 1,941 spam), 80/20 stratified split | Accuracy **98.2%**; spam precision **1.00**, spam recall **0.88** (388 spam in the 2,648-tweet test set) | A model that always says "ham" scores 85.3% accuracy with 0% spam recall, so accuracy alone is misleading. About 48 spam tweets were missed. Single split only. |
| 5-fold stratified CV on the same 13,240 tweets | Accuracy mean **98.2%** (98.0 to 98.3); spam recall mean **87.4%** (86.6 to 88.1) | Stable across folds. About 1 in 8 spam tweets is still missed. |
| `class_weight="balanced"`, same 5-fold CV | Accuracy 99.6%, spam recall 97.4%, spam precision 99.7% | Default model: 98.2% / 87.4% / 100%. |
| Same, keeping near-duplicate tweets in one fold (grouped by first 40 characters) | Accuracy 99.4%, spam recall 96.5%, spam precision 99.7% | Default model on the same split: 98.1% / 86.9% / 100%. The gain is real; near-duplicates inflate scores only slightly. Only 2 configurations were compared, so a small optimism remains. |
| `ngram_range=(1, 2)` with `class_weight="balanced"`, same 5-fold CV | Spam recall 96.7% | Not better than single words (97.4%). The 0.7-point gap is within the fold-to-fold spread (about 1.5 points), so it may be noise. Only recall was measured. |
### Error analysis
- On the first 80/20 split, 48 of 388 spam tweets were missed (12%). Missed spam is not shorter than caught spam (about 113 vs 105 characters).
- Reading 20 missed tweets by hand: most are adult content or ads for products and services (weight loss, health products, forex trading), written like normal conversational Arabic with no classic ad phrases. This is a small manual sample, so it is a hypothesis, not a proven cause.
- Exact duplicates: 260 extra copies (481 tweets involved), all labeled ham. Removing them changes nothing (accuracy 98.14%, spam recall 87.5%). Near-duplicates (almost identical text) were not checked. Update: a rough check (grouped by the first 40 characters) is in the results table above.
### Limitations
- The first dataset was tiny and hand-written; results were **unstable**.
- The larger test used tweets (a public dataset), not SMS or email, and a single train/test split. Classes are imbalanced (about 85% ham). The 5-fold CV above shows the results are stable across splits.
- Not production-ready.

### What I learned
- Accuracy on a tiny test set can be misleading; cross-validation gives a more honest estimate.
- With imbalanced classes, accuracy hides failures: always check precision and recall.
- Data quality (correct labels, in the right order) matters more than code.

---

## العربية

### نظرة عامة
مشروع صغير لمعالجة اللغة الطبيعية: تصنيف رسائل نصية عربية قصيرة إلى **سبام** أو **عادية**.
اتعمل كجزء من رحلة تعلّم تعلّم الآلة، باستخدام بايثون في `Google Colab`.

### النتائج (بأرقام صادقة)
| الاختبار | النتيجة | ملاحظة |
|---|---|---|
| تقسيم واحد 70/30 (9 رسايل) | 100 بالمئة | عدد قليل جدًا |
| 10 رسايل أصعب | 70 بالمئة | أخطاء: عادية اتصنّفت سبام |
| تقييم مكرر 5×5 (50 رسالة) | متوسط 87.2 بالمئة | تفاوت كبير |
| **داتا أكبر:** 13,240 تغريدة عربية (11,299 عادية / 1,941 سبام) | الدقة 98.2 بالمئة. دقة السبام `Precision` = 1.00. استرجاع السبام `Recall` = 0.ج88 | النموذج الغبي "الكل عادية" بيجيب 85.3 بالمئة دقة، وصفر بالمئة استرجاع للسبام، فالدقة لوحدها بتخدع. حوالي 48 تغريدة سبام عدّت |
| تقييم 5-fold (Cross-Validation) على نفس الـ 13,240 تغريدة | الدقة: متوسط 98.2 بالمئة (من 98.0 لـ 98.3). استرجاع السبام: متوسط 87.4 بالمئة(من 86.6 لـ 88.1) | ثابتة بين التقسيمات. تقريبًا تغريدة سبام من كل 8 بتعدّي. |
| تقييم 5-fold مع `class_weight="balanced"` | الدقة 99.6 بالمئة، استرجاع السبام 97.4 بالمئة، Precision السبام 99.7 بالمئة | النموذج الأصلي: 98.2 / 87.4 / 100. |
| نفس التقييم مع منع التغريدات المتشابهة (أول 40 حرف) من الانفصال بين التدريب والاختبار | الدقة 99.4 بالمئة، استرجاع السبام 96.5 بالمئة، Precision السبام 99.7 بالمئة | الأصلي على نفس التقسيم: 98.1 / 86.9 / 100. التحسين حقيقي، والتشابه الجزئي بيضخّم النتيجة شوية بس. قارنّا إعدادين بس. |
| تقييم `ngram_range=(1, 2)` مع `class_weight="balanced"`، نفس الـ 5-fold CV | recall السبام 96.7 بالمئة | مش أحسن من الكلمات المفردة (97.4). الفرق 0.7 نقطة جوه تذبذب التقسيمات (حوالي 1.5 نقطة)، فممكن يكون ضوضاء. قِسنا الـ recall بس. |
### تحليل الأخطاء
- في التقسيم الأول فات 48 من 388 تغريدة سبام (12 بالمئة). السبام اللي فات مش أقصر من اللي اتمسك (متوسط الطول حوالي 113 مقثابل 105 حرف).
- قرينا 20 تغريدة فاتت بإيدينا: أغلبها محتوى للكبار أو إعلانات منتجات وخدمات (تخسيس، علاج، فوركس)، مكتوبة بأسلوب كلام عادي من غير عبارات إعلانية واضحة. دي عيّنة صغيرة، فده تخمين مش سبب مثبت.
- التكرار بالنص بالظبط: 260 نسخة زيادة (481 تغريدة ليها توأم)، كلهم عادية (Ham). شيلهم ما غيّرش حاجة (الدقة 98.14 بالمئة، استرجاع السبام 87.5 بالمئة). التشابه الجزئي ما اتفحصش. تحديث: فحصناه بطريقة خشنة (أول 40 حرف)، وهو موضّح في الجدول فوق.
### الحدود
- الداتا الأولى قليلة ومكتوبة يدويًا، والنتائج كانت غير مستقرة.
- الاختبار الأكبر كان على تغريدات مش `SMS`، وبتقسيمة واحدة بس. **غير جاهز للاستخدام الفعلي**.

### اللي اتعلمته
- الدقة على اختبار صغير بتضلّل، والتقييم المتقاطع `Cross-Validation` أصدق.

- مع الأصناف غير المتوازنة، الدقة بتخبّي الفشل، فلازم نقيس معاها `Precision`
- ونقيس كمان `Recall`
