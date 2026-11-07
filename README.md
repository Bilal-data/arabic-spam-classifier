# Arabic Spam / Ham Text Classifier (Learning Project)
# مصنّف رسائل عربية: سبام / عادية (مشروع تعليمي)

---

## English

### Overview
A small end-to-end NLP project that classifies short Arabic text messages as **spam** or **ham** (normal).
Built as part of my machine-learning learning path, using Python in Google Colab.

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
| **Larger real dataset:** 13,240 Arabic tweets (11,299 ham / 1,941 spam), 80/20 stratified split | Accuracy **98.2%**; spam precision **1.00**, spam recall **0.88** (388 spam in the 2,648-tweet test set) | A model that always says "ham" scores 85.3% accuracy with 0% spam recall, so accuracy alone is misleading. About 46 spam tweets were missed. Single split only. |

### Limitations
- The first dataset was tiny and hand-written; results were **unstable**.
- The larger test used tweets (a public dataset), not SMS or email, and a single train/test split. Classes are imbalanced (about 85% ham).
- Not production-ready.

### What I learned
- Accuracy on a tiny test set can be misleading; cross-validation gives a more honest estimate.
- With imbalanced classes, accuracy hides failures: always check precision and recall.
- Data quality (correct labels, in the right order) matters more than code.

---
## العربية

### نظرة عامة
مشروع صغير لمعالجة اللغة الطبيعية: تصنيف رسائل نصية عربية قصيرة إلى **سبام** أو **عادية**.
اتعمل كجزء من رحلة تعلّم تعلّم الآلة، باستخدام بايثون في Google Colab.

### النتائج (بأرقام صادقة)
| الاختبار | النتيجة | ملاحظة |
|---|---|---|
| تقسيم واحد 70/30 (9 رسايل) | 100% | عدد قليل جدًا |
| 10 رسايل أصعب | 70% | أخطاء: عادية اتصنّفت سبام |
| Cross-Validation مكرر 5×5 (50 رسالة) | متوسط 87.2% | تفاوت كبير |
| **داتا أكبر:** 13,240 تغريدة عربية (11,299 عادية / 1,941 سبام) | الدقة **98.2%**؛ Precision للسبام **1.00**، وRecall للسبام **0.88** | النموذج الغبي "الكل عادية" بيجيب 85.3% دقة و0% Recall، فالدقة لوحدها بتخدع. حوالي 46 تغريدة سبام عدّت |

### الحدود
- الداتا الأولى قليلة ومكتوبة يدويًا، والنتائج كانت غير مستقرة.
- الاختبار الأكبر كان على تغريدات مش SMS، وبتقسيمة واحدة بس. **غير جاهز للاستخدام الفعلي**.

### اللي اتعلمته
- الدقة على اختبار صغير بتضلّل، والـ Cross-Validation أصدق.
- مع الأصناف غير المتوازنة، الدقة بتخبّي الفشل: لازم Precision وRecall.
