LANG2LANGKEY = {
    "Chinese": "zh",
    "Arabic": "ar",
    "Spanish": "es",
    "Japanese": "ja",
    "Hebrew": "he",
    "Turkish": "tr",
    "Ukrainian": "uk",
    "Yoruba": "yo",
    "Basque": "eu",
}

# Dimensions and language keys specific to the HiTZ/BASSE dataset
BASSE_DIMS = ["coherence", "consistency", "fluency", "relevance", "5W1H"]
BASSE_LANG2NAME = {"es": "Spanish", "eu": "Basque"}

LANG2PROMPT_coh = {
    "Chinese": "为以下中文文本生成前后连贯的摘要：{article}\n摘要：",
    "Arabic": "أنشئ ملخصًا متماسكًا للنص التالي باللغة العربية: {article}\nالملخص:",
    "Spanish": "Genere un resumen coherente del siguiente texto en español: {article}\nResumen:",
    "Japanese": "次の日本語のテキストの一貫した要约を生成します: {article}\n要约:",
    "Hebrew": "צור סיכום קוהרנטי לטקסט הבא בעברית: {article}\nסיכום:",
    "Turkish": "Aşağıdaki metnin Türkçe tutarlı bir özetini oluşturun: {article}\nÖzet:",
    "Ukrainian": "Створіть зв’язний анотація тексту українською мовою: {article}\nАнотація:",
    "Yoruba": "Ṣe apẹrẹ itọpọ fun ọrọ wọnyi ni ede Yoruba: {article}\nAkopọ:",
    "Basque": "Idatzi ondoko testuaren laburpen koherentea euskaraz: {article}\nLaburpena:",
}

LANG2PROMPT_com = {
    "Chinese": "为以下中文文本生成一段内容完整的摘要：{article}\n摘要：",
    "Arabic": "أنشئ ملخصًا كاملاً للنص التالي باللغة العربية: {article}\nالملخص:",
    "Spanish": "Genere un resumen completo del siguiente texto en español: {article}\nResumen:",
    "Japanese": "次の日本語テキストの完全な要约を生成します: {article}\n要约:",
    "Hebrew": "צור תקציר מלא לטקסט הבא בעברית: {article}\n תקציר:",
    "Turkish": "Aşağıdaki metnin Türkçe tam özetini oluşturun: {article}\nÖzet:",
    "Ukrainian": "Створіть повний виклад наступного тексту українською мовою: {article}\nВиклад:",
    "Yoruba": "Kọ akojọpọ pipe fun ọrọ wọnyi ni ede Yoruba: {article}\nAkopọ:",
    "Basque": "Idatzi ondoko testuaren laburpen osoa euskaraz: {article}\nLaburpena:",
}

# ── New BASSE dimensions (Spanish + Basque) ──────────────────────────────────

LANG2PROMPT_con = {
    "Spanish": "Genera un resumen fiel al contenido del siguiente artículo en español: {article}\nResumen:",
    "Basque": "Idatzi ondoko artikuluari leial den laburpena euskaraz: {article}\nLaburpena:",
}

LANG2PROMPT_flu = {
    "Spanish": "Genera un resumen fluido y bien redactado del siguiente artículo en español: {article}\nResumen:",
    "Basque": "Idatzi ondoko artikuluaren laburpen jario onekoa euskaraz: {article}\nLaburpena:",
}

LANG2PROMPT_rel = {
    "Spanish": "Genera un resumen con la información más relevante del siguiente artículo en español: {article}\nResumen:",
    "Basque": "Idatzi ondoko artikuluaren informaziorik garrantzitsuena jasotzen duen laburpena euskaraz: {article}\nLaburpena:",
}

LANG2PROMPT_5w1h = {
    "Spanish": "Genera un resumen con quién, qué, cuándo, dónde, por qué y cómo del siguiente artículo en español: {article}\nResumen:",
    "Basque": "Idatzi nor, zer, noiz, non, zergatik eta nola aipatzen dituen laburpena ondoko artikuluarentzat euskaraz: {article}\nLaburpena:",
}

LANG2PROMPT_task_intro = {
    "Chinese": "您将收到一篇新闻文章的摘要。您的任务是根据一项指标对摘要进行评分。请务必仔细阅读并理解这些说明。请在审阅过程中保持此文档打开，并在需要时参考它。",
    "Arabic": "ستتلقى ملخصًا لمقال إخباري. مهمتك هي تقييم الملخص بناءً على مقياس واحد. يُرجى قراءة هذه التعليمات وفهمها بعناية. احتفظ بهذه الوثيقة مفتوحة أثناء عملية المراجعة، وراجعها عند الحاجة.",
    "Spanish": "Recibirás un resumen de un artículo periodístico. Tu tarea consiste en puntuar dicho resumen según un criterio. Lee y comprende atentamente estas instrucciones. Mantén este documento abierto durante el proceso de revisión y consúltalo cuando lo necesites.",
    "Japanese": "ニュース記事の要約を受け取ります。課題は、1つの指標に基づいて要約を採点することです。以下の指示をよく読んで理解してください。レビュープロセス中は、この文書を開いたままにして、必要に応じて参照してください。",
    "Hebrew": "תקבלו סיכום של כתבה חדשותית. המשימה שלכם היא לדרג את הסיכום על סמך מדד אחד. אנא קראו והבינו את ההוראות בעיון. שמרו על מסמך זה פתוח במהלך תהליך הבדיקה ועיינו בו בעת הצורך.",
    "Turkish": "Bir haber makalesinin özetini alacaksınız. Göreviniz, özeti bir ölçüte göre puanlamaktır. Lütfen bu talimatları dikkatlice okuyun ve anlayın. İnceleme süreci boyunca bu belgeyi açık tutun ve gerektiğinde başvurun.",
    "Ukrainian": "Ви отримаєте короткий виклад новинної статті. Ваше завдання — оцінити короткий виклад на основі одного показника. Будь ласка, уважно прочитайте та зрозумійте ці інструкції. Тримайте цей документ відкритим під час процесу рецензування та звертайтеся до нього за потреби.",
    "Yoruba": "A o gba akopọ iroyin kan. Iṣẹ́ rẹ ni lati gba ami akopọ ti a da lori iwọn kan. Jọwọ ka ki o si loye awọn ilana wọnyi daradara. Pa iwe yii mọ ni ṣiṣi lakoko ilana atunyẹwo ki o si tọka si nigbati o ba nilo.",
    # You will be given one summary written for a news article. Your task is to rate the summary on one metric. Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.
    "Basque": "Albiste-artikulu baten laburpena jasoko duzu. Zure zeregina laburpena adierazle baten arabera puntuatzea da. Arretaz irakurri eta ulertu jarraibide hauek. Mantendu dokumentu hau irekita berrikuspena egiten duzun bitartean eta behar denean kontsultatu.",
}

LANG2PROMPT_coh_criteria = {
    "Chinese": "评估标准：\n连贯性（1-5分）——所有句子的整体质量。摘要应结构清晰、组织良好。摘要不应只是相关信息的堆砌，而应逐句构建，最终形成关于某一主题的连贯信息。\n",
    "Arabic": "\nمعايير التقييم: الترابط (١-٥) - الجودة الجماعية لجميع الجمل. يجب أن يكون الملخص منظمًا وذا بنية جيدة. لا ينبغي أن يكون الملخص مجرد كومة من المعلومات المترابطة، بل يجب أن يبني من جملة إلى جملة ليشكل مجموعة متماسكة من المعلومات حول موضوع معين.",
    "Spanish": "Criterios de evaluación:\n Coherencia (1-5) - Calidad general de todas las oraciones. El resumen debe estar bien organizado y estructurado. El resumen no debe ser una mera recopilación de información relacionada, sino que debe desarrollarse oración por oración para formar un conjunto coherente de información sobre un tema específico.\n",
    "Japanese": "評価基準：\n一貫性（1～5） - すべての文の全体的な質。 要約は、よく整理され、構造化されている必要があります。要約は、関連情報を単に集めたものではなく、文ごとに積み重ねられ、特定のトピックに関する一貫した情報体系を形成する必要があります。\n",
    "Hebrew": "\nקריטריונים להערכה: קוהרנטיות (1-5) - האיכות הכוללת של כל המשפטים. הסיכום צריך להיות מאורגן ומובנה היטב. הסיכום לא צריך להיות אוסף מידע קשור בלבד, אלא צריך לבנות על עצמו משפט אחר משפט כדי ליצור גוף מידע קוהרנטי בנושא מסוים.",
    "Turkish": "Değerlendirme Kriterleri:\n Tutarlılık (1-5) - Tüm cümlelerin genel kalitesi. Özet iyi organize edilmiş ve yapılandırılmış olmalıdır. Özet, yalnızca ilgili bilgilerin bir araya getirilmesinden oluşmamalı, belirli bir konu hakkında tutarlı bir bilgi bütünü oluşturmak için cümle cümle kendi üzerine inşa edilmelidir.\n",
    "Ukrainian": "Критерії оцінювання:\n Зв'язність (1-5) - Загальна якість усіх речень. Анотація має бути добре організованою та структурованою. Анотація не повинна бути простою колекцією пов'язаної інформації, а радше повинна будуватися речення за реченням, формуючи цілісний корпус інформації з певної теми.\n",
    "Yoruba": "Àwọn Ìlànà Ìṣàyẹ̀wò:\n Ìbáramu (1-5) - Dídára gbogbo gbogbo gbólóhùn. Àkópọ̀ náà yẹ kí ó wà ní ìṣètò dáadáa àti ní ìṣètò. Àkópọ̀ náà kò gbọdọ̀ jẹ́ àkójọpọ̀ ìsọfúnni tí ó jọra lásán, ṣùgbọ́n ó yẹ kí ó kọ́ gbólóhùn kọ̀ọ̀kan láti ṣẹ̀dá àkójọpọ̀ ìsọfúnni tí ó ṣọ̀kan lórí kókó kan pàtó.\n",
    # Evaluation Criteria: Coherence (1-5) - The overall quality of all sentences. The summary should be well-organized and structured. The summary should not be a mere collection of related information, but rather should build upon itself sentence by sentence to form a coherent body of information on a specific topic.
    "Basque": "Ebaluazio irizpideak:\nKoherentzia (1-5) - Esaldi guztien kalitate orokorra. Laburpenak ondo antolatua eta egituratua egon behar du. Laburpena ez da informazio erlazionatuaren bilketa hutsa izan behar, esaldiz esaldi eraiki behar da gai jakin baten inguruko informazio koherente bat osatzeko.\n",
}

LANG2PROMPT_com_criteria = {
    "Chinese": "评估标准：\n完整性（1-5分）——摘要涵盖新闻文章所有重要信息的程度，既不遗漏关键细节，也不添加无关内容。\n",
    "Arabic": "معايير التقييم: الاكتمال (1-5) - الدرجة التي يغطي بها الملخص جميع المعلومات المهمة من مقال الأخبار دون إغفال التفاصيل الرئيسية أو إضافة محتوى غير ذي صلة.",
    "Spanish": "Criterios de evaluación:\nIntegridad (1–5) – el grado en que el resumen abarca toda la información importante del artículo de noticias sin omitir detalles clave ni añadir contenido irrelevante.\n",
    "Japanese": "評価基準:\n完全性（1～5）– 要約が、重要な詳細を省略したり無関係な内容を追加したりすることなく、ニュース記事のすべての重要な情報を網羅している程度。\n",
    "Hebrew": "קריטריונים להערכה:\nשלמות (1-5) – המידה שבה הסיכום מכסה את כל המידע החשוב מהכתבה החדשותית מבלי להשמיט פרטים מרכזיים או להוסיף תוכן לא רלוונטי.\n",
    "Turkish": "Değerlendirme Kriterleri:\nTamlık (1–5) – özetin, haber makalesindeki tüm önemli bilgileri, önemli ayrıntıları atlamadan veya alakasız içerik eklemeden kapsama derecesi.\n",
    "Ukrainian": "Критерії оцінювання:\nПовнота (1–5) – ступінь, до якої короткий виклад охоплює всю важливу інформацію з новинної статті, не пропускаючи ключових деталей або не додаючи зайвого контенту.\n",
    "Yoruba": "Àwọn Ìlànà Ìṣàyẹ̀wò:\nPípé (1–5) – ìwọ̀n tí àkópọ̀ náà fi bo gbogbo ìwífún pàtàkì láti inú àpilẹ̀kọ ìròyìn láìsí pé a yọ àwọn kúlẹ̀kúlẹ̀ pàtàkì kúrò tàbí a fi àwọn àkóónú tí kò ṣe pàtàkì kún un.\n",
    # "Evaluation Criteria:\nCompleteness (1–5) – the degree to which the summary covers all important information from the news article without omitting key details or adding irrelevant content.\n"
    "Basque": "Ebaluazio irizpideak:\nOsotasuna (1–5) – laburpenak albiste-artikuluko informazio garrantzitsu guztia zein neurritan biltzen duen, funtsezko xehetasunak ez baztertuta eta irrelevantea den edukirik gehitu gabe.\n",
}

# G-eval criteria for the three additional BASSE dimensions (Spanish + Basque)

LANG2PROMPT_con_criteria = {
    "Spanish": "Criterios de evaluación:\nConsistencia (1-5) – el grado en que el resumen es factualmente consistente con el artículo fuente. Un resumen consistente solo contiene afirmaciones respaldadas por el artículo fuente.\n",
    "Basque": "Ebaluazio irizpideak:\nKoherentzia faktikoa (1-5) – laburpena iturri-artikuluarekin zenbateraino den koherente faktikoki. Laburpen koherenteak iturri-artikuluak babestutako adierazpenak bakarrik ditu.\n",
}

LANG2PROMPT_flu_criteria = {
    "Spanish": "Criterios de evaluación:\nFluidez (1-5) – la calidad del resumen en términos de gramática, legibilidad y naturalidad. Una puntuación alta indica que el resumen está bien escrito y es fácil de leer.\n",
    "Basque": "Ebaluazio irizpideak:\nJariakortasuna (1-5) – laburpenaren kalitatea gramatika, irakurgarritasuna eta naturalartasunaren aldetik. Puntuazio altu batek laburpena ondo idatzita dagoela eta irakurtzeko erraza dela adierazten du.\n",
}

LANG2PROMPT_rel_criteria = {
    "Spanish": "Criterios de evaluación:\nRelevancia (1–5) – el grado en que el resumen captura la información más importante del artículo fuente sin incluir información innecesaria.\n",
    "Basque": "Ebaluazio irizpideak:\nGarrantzia (1–5) – laburpenak iturri-artikuluaren informaziorik garrantzitsuena zenbateraino jasotzen duen informazio alferrikakoa sartu gabe.\n",
}

LANG2PROMPT_5w1h_criteria = {
    "Spanish": "Criterios de evaluación:\n5W1H (1–5) – el grado en que el resumen responde a las preguntas quién, qué, cuándo, dónde, por qué y cómo del artículo fuente.\n",
    "Basque": "Ebaluazio irizpideak:\n5N1Z (1–5) – laburpenak nor, zer, noiz, non, zergatik eta nola galderak iturri-artikuluan oinarrituz zenbateraino erantzuten dituen.\n",
}

LANG2PROMPT_article_cache = {
    "Chinese": "原文：{article}",
    "Arabic": "المقال الأصلي:{article}",
    "Spanish": "Artículo original: {article}",
    "Japanese": "元記事：{article}",
    "Hebrew": "מאמר מקורי：{article}",
    "Turkish": "Orijinal makale:{article}",
    "Ukrainian": "Оригінальна стаття：{article}",
    "Yoruba": "Àpilẹ̀kọ àtilẹ̀wá:{article}",
    "Basque": "Jatorrizko artikulua: {article}",
}

LANG2PROMPT_article_summary = {
    "Chinese": "原文：{article} \n摘要：{summary}\n你的評分：",
    "Arabic": "المقال الأصلي：{article}\nالملخص：{summary}\nتقييمك：",
    "Spanish": "Artículo original: {article}\nResumen: {summary}\nTu valoración:",
    "Japanese": "元記事：{article}\n概要：{summary}\nあなたの評価：",
    "Hebrew": "מאמר מקורי：{article}\nסיכום：{summary}\nהדירוג שלך：",
    "Turkish": "Orijinal makale:{article}\nÖzet:{summary}\nPuanınız:",
    "Ukrainian": "Оригінальна стаття：{article}\nКороткий зміст：{summary}\nВаша оцінка：",
    "Yoruba": "Àpilẹ̀kọ àtilẹ̀wá:{article}\nÀkótán:{summary}\nÌwọ̀n rẹ:",
    "Basque": "Jatorrizko artikulua: {article}\nLaburpena: {summary}\nZure balorazioa:",
}

# Aliases used by intervention_geval.py
LANG2PROMPT_article_cache_geval = LANG2PROMPT_article_cache
LANG2PROMPT_article_summary_geval = LANG2PROMPT_article_summary

MAX_LENGTH = 1024

LANGS = [
    "Chinese",
    "Arabic",
    "Spanish",
    "Japanese",
    "Hebrew",
    "Turkish",
    "Ukrainian",
    "Yoruba",
]

SIGMAS_BY_MODE = {
    "baseline": [0.0],
    "vector": [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    "map": [0.0, 0.25, 0.5, 0.75, 1.0],
}

MAX_PAIRS = 500
