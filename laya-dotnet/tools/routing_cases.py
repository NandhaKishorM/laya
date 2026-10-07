"""Probe corpus for the routing/language/email/shortlist golden vectors.

Consumed only by `laya-dotnet/tools/dump_routing_golden.py`, which runs every entry through the real
`laya.lang` / `laya.router` / `laya.email` / `laya.shortlist` code paths and records
whatever Python actually produces. Nothing in this module is itself a golden value -- it is
the *input* side only, chosen to hit the parity pitfalls documented in the porting plan
(Unicode code-point iteration, `str.lower()`/`isalpha()` semantics, `max()` tie order,
`round()` half-even, quote/signature/disclaimer regexes, cosine edge cases).

Four sections, one per probe file:
    LANG_STATES      -> lang_probe.json      (lang.analyse / detect_script / script_profile /
                                               latin_profile / state_text)
    ROUTE_CASES       -> route_probe.json     (router.Router.route)
    EMAIL_CASES        -> email_probe.json    (email.clean_email_body / email_state)
    SHORTLIST_CASES    -> shortlist_probe.json (shortlist.shortlist_choice)

Plus the single source of truth for the combined routing sample's own fixed inputs
(SAMPLE_* below), so the samples can be diffed against a Python-recorded
golden the same way the existing `Laya.Sample` is.
"""

# ============================================================================ lang_probe
#
# label -> state, fed to `lang.analyse(state)` and `lang.state_text(state)` directly, and to
# `detect_script`/`script_profile`/`latin_profile` via `state_text(state)` (dump script's job,
# not this module's -- these three take plain text, not a full state).

LANG_STATES = [
    ("upstream/swedish_short", "glömt lösenord"),
    ("upstream/swedish_login", "kan inte logga in"),
    ("upstream/nordic_overlap", "hej min vän"),
    ("upstream/banglish", "ami amar taka ferot chai"),
    ("upstream/azerbaijani", "mən bu hesab üçün kömək istəyirəm"),
    ("upstream/german_ascii", "ich habe mein paket noch nicht"),
    ("upstream/portuguese_ascii", "eu nao consigo entrar na minha conta"),
    ("upstream/english_loanword", "Please send me the café receipt."),
    ("upstream/repeated_collision", "come come come please send the receipt"),
    ("upstream/identifier_not_prose", "Please check my error at os.path and user@example.com"),
    ("upstream/fullwidth_latin", "ＡＢＣＤ hello"),
    ("upstream/ipa_latin", "ɐɪɲ hello"),
    ("upstream/cjk_brand", "iPhone 用户无法登录账户请帮助"),
    ("upstream/cyrillic_name", "Please send the receipt to Владимир Иванов today"),
    ("upstream/greek_symbol", "Please set α to the correct value"),
    ("upstream/mixed_foreign_line",
     "Please review the error and send me the details.\nEu nao consigo entrar na minha conta."),
    ("upstream/mixed_foreign_field",
     {"note": "Please review the error and send me the details. " * 100,
      "body": "ich habe mein paket noch nicht"}),
    ("upstream/mixed_unknown_latin_field",
     {"note": "Please review the error and send me the details. " * 100,
      "body": "žluťoučký kůň úpěl ďábelské ódy"}),
    ("upstream/mixed_nonlatin_field",
     {"note": "Please review the error and send me the details. " * 100,
      "body": "我的账户无法登录请帮助我"}),
    ("upstream/mixed_code_line",
     "Please review the error and send me the details.\nos.path = nao consigo entrar na minha conta"),
    # ---- straight from tests/test_router.py SCRIPTS (script detection) ----
    ("script/english", "The customer was charged twice and wants a refund."),
    ("script/armenian", "Հայերեն"),
    ("script/armenian_uppercase", "ՀԱՅԵՐԵՆ"),
    ("script/armenian_punctuation_only", "։֊"),
    ("script/french", "Le client a été facturé deux fois et demande un remboursement."),
    ("script/hindi", "ग्राहक से दो बार शुल्क लिया गया और वह धनवापसी चाहता है।"),
    ("script/japanese", "お客様は二重に請求されたため返金を希望しています。"),
    ("script/chinese", "客户被重复扣款要求退款"),
    ("script/korean", "고객이 두 번 청구되어 환불을 원합니다"),
    ("script/arabic", "تم خصم المبلغ مرتين من العميل ويريد استرداد الأموال"),
    ("script/tamil", "வாடிக்கையாளரிடம் இருமுறை கட்டணம் வசூலிக்கப்பட்டது"),
    ("script/russian", "С клиента дважды сняли деньги и он хочет возврат"),
    ("script/thai", "ลูกค้าถูกเรียกเก็บเงินสองครั้งและต้องการเงินคืน"),
    ("script/greek", "Ο πελάτης χρεώθηκε δύο φορές και θέλει επιστροφή χρημάτων"),
    ("script/hebrew", "הלקוח חויב פעמיים ורוצה החזר כספי"),
    ("script/empty", ""),
    ("script/digits_only", "12345 6789"),

    # ---- straight from tests/test_router.py is_english checks ----
    ("is_english/plain_english", "Please refund the duplicate charge on invoice 4411 today."),
    ("is_english/armenian", "Հայերեն"),
    ("is_english/english_short", "refund me"),
    ("is_english/hindi", "ग्राहक से दो बार शुल्क लिया गया"),
    ("is_english/japanese", "お客様は二重に請求されました"),
    ("is_english/russian", "С клиента дважды сняли деньги"),
    ("is_english/french_long", "Le client a été facturé deux fois et il demande un remboursement pour la "
                                "facture qui a été payée le mois dernier avec la carte de crédit"),
    ("is_english/german_long", "Der Kunde wurde zweimal belastet und möchte eine Rückerstattung für die "
                                "Rechnung die nicht korrekt ist und auch nicht bezahlt wurde"),
    ("is_english/romanian", "Gătește-mi o rețetă de sarmale de post pentru mâine."),
    ("is_english/romanian_invoice", "Am fost taxat de două ori pentru factura din luna martie și vreau banii"),
    ("is_english/polish", "Klient został obciążony dwukrotnie i chce zwrot pieniędzy za fakturę"),
    ("is_english/czech", "Zákazníkovi byla částka účtována dvakrát a žádá o vrácení peněz"),
    ("is_english/turkish", "Müşteriden iki kez ücret alındı ve para iadesi istiyor lütfen yardım"),
    ("is_english/vietnamese", "Khách hàng đã bị thu phí hai lần và muốn được hoàn tiền ngay"),
    ("is_english/english_with_loanwords", "We visited a cafe in Zurich and the naive assumption about the "
                                           "invoice was wrong, so please refund the duplicate charge"),
    ("is_english/undecided_turkish", "Müşteriden iki kez ücret alındı ve para iadesi istiyor"),
    ("is_english/english_not_undecided", "Please refund the duplicate charge on the invoice"),
    ("is_english/romanian_diacritics", "Gătește-mi o rețetă de sarmale"),
    ("is_english/no_diacritics", "Please refund the duplicate charge today"),
    ("is_english/zero_tie_romanian", "Cât e ora acum la Tokyo"),
    ("is_english/known_gap_romanian_no_diacritics", "Care este ora in Tokyo?"),

    # ---- latin language guess ----
    ("latin_lang/english", "The customer was charged twice and wants a refund for this invoice"),
    ("latin_lang/french", "Le client a ete facture deux fois et il demande un remboursement pour la facture"),
    ("latin_lang/german", "Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung fuer die Rechnung"),
    ("latin_lang/spanish", "El cliente fue cobrado dos veces y quiere que le devuelvan el dinero por la factura"),
    ("latin_lang/too_short", "refund"),
    ("latin_lang/long_english_stays_en",
     "Please refund the duplicate charge on invoice 4411 today because we have been waiting "
     "for three days and nobody has replied to us"),

    # ---- state flattening ----
    ("state_text/dict", {"body": "charged twice", "n": 3}),
    ("state_text/nested", {"a": {"b": ["deep"]}}),
    ("state_text/list", ["x", {"y": "z"}]),
    ("state_text/none", None),
    ("state_text/keys_ignored", {"subject": "नमस्ते", "body": "ग्राहक से दो बार शुल्क लिया गया"}),

    # ---- workflow/profile edge cases from tests ----
    ("profile/armenian_pure", "Հայերեն"),
    ("profile/armenian_mixed_with_latin", "Հայերեն abc"),

    # ---- plain-ASCII Romance (#172): accent-stripped states must still route non-English ----
    ("plain_ascii/es_1", "El pedido llego roto y nadie responde cuando escribo al soporte"),
    ("plain_ascii/es_2", "Quiero cancelar mi plan y pedir un reembolso"),
    ("plain_ascii/es_3", "La factura tiene un error en el importe total"),
    ("plain_ascii/es_4", "Necesito que me devuelvan el dinero de la compra duplicada"),
    ("plain_ascii/it_1", "Il cliente e stato addebitato due volte e vuole un rimborso"),
    ("plain_ascii/it_2", "Voglio cancellare il mio abbonamento e chiedere un rimborso"),
    ("plain_ascii/it_3", "La fattura contiene un errore nell importo totale"),
    ("plain_ascii/pt_1", "O cliente foi cobrado duas vezes e quer o dinheiro de volta"),
    ("plain_ascii/fr_1", "Le client a ete facture deux fois et demande un remboursement"),
    ("plain_ascii/fr_2", "Je ne peux pas acceder a mon compte et j ai besoin d aide"),
    ("accented/es", "La facturación tiene un error y necesito una corrección urgente"),
    ("accented/it", "La fattura è sbagliata, devo avere un rimborso per il pagamento"),
    ("accented/fr", "La commande est arrivée cassée et personne ne répond au support"),
    ("romance_control/1", "The customer was charged twice and wants a refund for this invoice"),
    ("romance_control/2", "Please cancel my subscription and refund the duplicate charge today"),
    ("romance_control/3", "The report by Smith et al. shows the de facto standard, e.g. the LA office and Rio"),
    ("romance_control/4", "Our MI5 and UN contacts discussed the DOS attack in LA last month"),
    ("romance_control/5", "No refund was issued, so I am writing to you again about invoice 4411"),
    ("romance_control/6", "no refund no reply"),
    ("romance_control/7", "The son of the director filed a complaint about the duplicate invoice"),
    ("shared_words/names_nothing", "Cât e ora acum la Tokyo"),
    ("shared_words/distinctive_names_it", "La fattura contiene un errore nell importo totale"),
    ("shared_words/shared_hits_still_count_es", "La factura tiene un error en el importe total"),

    # ---- all 25 non-Latin scripts recognised by _SCRIPT_RANGES, plus a couple more Latin ----
    ("all_scripts/greek", "Γεια σου κόσμε, πώς είσαι σήμερα;"),
    ("all_scripts/cyrillic", "Привет, как ты себя чувствуешь сегодня?"),
    ("all_scripts/armenian", "Բարեւ ձեզ, ինչպե՞ս եք այսօր"),
    ("all_scripts/hebrew", "שלום, מה שלומך היום"),
    ("all_scripts/arabic", "مرحبا كيف حالك اليوم"),
    ("all_scripts/devanagari", "नमस्ते आप कैसे हैं आज"),
    ("all_scripts/bengali", "আমি ভালো আছি, আপনি কেমন আছেন"),
    ("all_scripts/gurmukhi", "ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ, ਮੈਂ ਠੀਕ ਹਾਂ"),
    ("all_scripts/gujarati", "તમે કેમ છો, હું સારો છું"),
    ("all_scripts/oriya", "ତୁମେ କେମିତି ଅଛ, ମୁଁ ଭଲ ଅଛି"),
    ("all_scripts/tamil", "நீங்கள் எப்படி இருக்கிறீர்கள்"),
    ("all_scripts/telugu", "మీరు ఎలా ఉన్నారు, నేను బాగున్నాను"),
    ("all_scripts/kannada", "ನೀವು ಹೇಗಿದ್ದೀರಿ, ನಾನು ಚೆನ್ನಾಗಿದ್ದೇನೆ"),
    ("all_scripts/malayalam", "നിങ്ങൾക്ക് സുഖമാണോ, എനിക്ക് സുഖമാണ്"),
    ("all_scripts/sinhala", "ඔබට කොහොමද, මට හොඳින්"),
    ("all_scripts/thai", "คุณเป็นอย่างไรบ้างวันนี้"),
    ("all_scripts/lao", "ສະບາຍດີ ທ່ານເປັນແນວໃດແດ່"),
    ("all_scripts/tibetan", "བཀྲ་ཤིས་བདེ་ལེགས་ཁྱེད་རང་ག་འདྲ་ཡོད"),
    ("all_scripts/myanmar", "မင်္ဂလာပါ သင်နေကောင်းလား"),
    ("all_scripts/georgian", "გამარჯობა როგორ ხარ დღეს"),
    ("all_scripts/ethiopic", "ሰላም እንዴት ነህ ዛሬ"),
    ("all_scripts/khmer", "សួស្តី តើអ្នកសុខសប្បាយទេ"),
    ("all_scripts/hangul", "안녕하세요 오늘 기분이 어때요"),
    ("all_scripts/kana", "こんにちは、今日は元気ですか"),
    ("all_scripts/han", "你好，你今天过得怎么样"),
    ("all_scripts/latin", "Hello, how are you feeling today?"),

    # ---- mixed scripts / ties (max() tie order: first-inserted key wins; latin is always
    #      inserted last, so it never wins a tie against a script that appeared) ----
    ("tie/greek_then_cyrillic_1v1", "ΑА"),          # expect "greek" (first seen)
    ("tie/cyrillic_then_greek_1v1", "АΑ"),          # expect "cyrillic" (first seen)
    ("tie/han_then_latin_1v1", "中 a"),                        # expect "han" (latin inserted last)
    ("tie/latin_then_han_1v1", "a 中"),                        # expect "han" still (order in text ignored for latin)
    ("tie/three_way_greek_cyrillic_armenian", "ΑАԱ"),  # 1-1-1, expect "greek" (first)
    ("mixed/latin_han_arabic", "invoice 中文 فاتورة refund"),
    ("mixed/devanagari_latin_majority_devanagari", "ग्राहक से दो बार शुल्क लिया गया refund"),
    ("mixed/latin_majority_devanagari_minor", "Please refund the customer today ग्राहक"),

    # ---- latin_profile ties: _STOP dict order is en, fr, de, es, pt, it, nl, ro; a tied
    #      score picks the first-iterated language among those with real (non-shared) evidence ----
    ("latin_tie/fr_vs_de", "être cette nicht sich"),           # expect "fr" (fr before de)
    ("latin_tie/de_vs_es", "nicht sich el los"),                # expect "de" (de before es)
    ("latin_tie/es_vs_pt", "está una dos tres o não está"),
    ("latin_tie/pt_vs_it", "não está o della sono"),

    # ---- astral characters (outside the BMP): emoji is not alphabetic and is dropped from
    #      every count; a letter outside every _SCRIPT_RANGES bucket (Deseret, astral) is
    #      alphabetic but also silently dropped from counts (neither latin nor any script) ----
    ("astral/emoji_only", "\U0001F600\U0001F680\U00002764\U0000FE0F"),
    ("astral/emoji_with_latin", "refund please \U0001F600 today"),
    ("astral/deseret_letter_alone", "\U00010400\U00010401\U00010402"),   # Deseret, no matching script bucket
    ("astral/deseret_mixed_with_latin", "refund \U00010400 today"),
    ("astral/mathematical_bold_letters", "\U0001D400\U0001D401\U0001D402"),  # Mathematical Bold A B C

    # ---- combining marks: Python str.isalpha() is Unicode category L* only, so a combining
    #      mark (Mn) like the Devanagari virama is NOT counted as alphabetic; a port whose
    #      "is letter" check includes marks (e.g. Rust char::is_alphabetic) will disagree ----
    ("combining/devanagari_with_virama", "ग्राहक"),   # contains U+094D VIRAMA (Mn)
    ("combining/latin_with_combining_acute", "café résumé"),  # decomposed é

    # ---- Turkish dotted capital I: str.lower() maps İ -> 2 code points ("i" + combining dot) ----
    ("turkish_i/capital_i_with_dot", "İstanbul"),
    ("turkish_i/mixed_case", "İYİ GÜNLER İstanbul'da"),
    ("turkish_i/lowercase_dotless", "ıstanbul"),

    # ---- digits / numerics: `\w` (used by `_WORD`) = isalnum() + '_', which is letters and
    #      numeric No/Nl categories, not marks; digits (Nd) are explicitly excluded by `\d` ----
    ("numerics/ascii_digits_only", "12345 67890"),
    ("numerics/roman_numeral_nl", "chapter Ⅷ begins"),   # U+2167 ROMAN NUMERAL EIGHT (Nl)
    ("numerics/fullwidth_digits", "１２３ refund"),   # fullwidth 123 (Nd)
    ("numerics/mixed_digits_and_words", "invoice 4411 was paid twice on 2024-01-05"),

    # ---- nested dict/list states deeper than the _iter_text depth-6 cutoff ----
    ("nesting/depth_6_kept", {"a": {"b": {"c": {"d": {"e": {"f": "visible at depth 6"}}}}}}),
    ("nesting/depth_7_dropped", {"a": {"b": {"c": {"d": {"e": {"f": {"g": "invisible past depth 6"}}}}}}}),
    ("nesting/list_depth_7_dropped", [[[[[[["invisible list depth 7"]]]]]]]),
    ("nesting/mixed_depth_boundary",
     {"a": {"b": {"c": {"d": {"e": ["kept depth 6", {"f": "dropped depth 7"}]}}}}}),

    # ---- >4000-char inputs: state_text's default max_chars=4000 slice is per code point ----
    ("long/ascii_over_4000",
     "refund please. " * 400),  # 16 * 400 = 6400 chars
    ("long/multibyte_over_4000",
     "退款请求 " * 1000),  # 5 chars * 1000 = 5000 code points, multi-byte in UTF-8
    ("long/mixed_script_over_4000_boundary_near_switch",
     ("a" * 3990) + "客户被重复扣款要求退款" * 5),  # ascii run crosses the 4000 cut into Han

    # ---- English control long enough to stay decisively English despite length ----
    ("long/english_over_1000",
     "Please refund the duplicate charge on invoice 4411 today. " * 20),
]


# ============================================================================ route_probe
#
# Each entry: (label, router_kwargs, route_kwargs). `router_kwargs` builds the `Router`
# (state that survives across the whole case: `default`, `auto_task_detection`,
# `standalone_repos`, `models` override); `route_kwargs` is passed straight to `.route(...)`
# (state, questions, model=, task=, lang=). Router() with no models loads nothing -- `.route`
# never touches the network or the filesystem.

Q_GENERIC = {"dept": {"type": "choice", "instructions": "Which team?",
                       "criteria": {"billing": None, "tech": None}}}

_TD_IDS = {
    "agent_trace_observability": ["action", "needs_review", "outcome", "risk", "urgency"],
    "customer_service": ["action", "category", "churn_risk", "needs_human", "urgency"],
    "invoice_processing": ["discrepancy_severity", "disposition", "duplicate", "matches_order", "urgency"],
    "security_incidents": ["credential_compromise", "disposition", "severity", "true_positive", "urgency"],
}


def _td_questions(ids):
    return {i: {"type": "noul", "instructions": "x"} for i in ids}


ROUTE_CASES = [
    ("upstream/undetermined_latin_custom_default", {"default": "multilingual"}, {"state": "Hello"}),
    ("upstream/mixed_foreign_line", {}, {"state":
     "Please review the error and send me the details.\nEu nao consigo entrar na minha conta."}),
    # ---- default router, precedence ladder ----
    ("english_text", {}, {"state": {"body": "I was charged twice, please refund."}, "questions": Q_GENERIC}),
    ("armenian_text", {}, {"state": {"body": "Հայերեն"}, "questions": Q_GENERIC}),
    ("armenian_explicit_override", {}, {"state": {"body": "Հայերեն"}, "questions": Q_GENERIC, "model": "english"}),
    ("hindi_text", {}, {"state": {"body": "मुझसे दो बार शुल्क लिया गया"}, "questions": Q_GENERIC}),
    ("japanese_text", {}, {"state": {"body": "二重に請求されました"}, "questions": Q_GENERIC}),
    ("korean_text", {}, {"state": {"body": "두 번 청구되었습니다"}, "questions": Q_GENERIC}),
    ("arabic_text", {}, {"state": {"body": "تم خصم المبلغ مرتين"}, "questions": Q_GENERIC}),
    ("german_text", {}, {"state": {"body": "Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung "
                                            "fuer die Rechnung die nicht korrekt ist"}, "questions": Q_GENERIC}),
    ("explicit_model", {}, {"state": {"body": "anything"}, "questions": Q_GENERIC, "model": "multilingual"}),
    ("explicit_model_overrides_script", {}, {"state": {"body": "मुझसे दो बार"}, "questions": Q_GENERIC, "model": "english"}),
    ("explicit_task", {}, {"state": {"body": "x"}, "questions": Q_GENERIC, "task": "typed_decisions"}),
    ("explicit_lang_en", {}, {"state": {"body": "मुझसे दो बार"}, "questions": Q_GENERIC, "lang": "en"}),
    ("explicit_lang_de", {}, {"state": {"body": "hello there"}, "questions": Q_GENERIC, "lang": "de"}),
    ("explicit_lang_eng_alias", {}, {"state": {"body": "hello"}, "questions": Q_GENERIC, "lang": "eng"}),
    ("explicit_lang_english_word", {}, {"state": {"body": "hello"}, "questions": Q_GENERIC, "lang": "English"}),
    ("explicit_lang_en_us", {}, {"state": {"body": "hello"}, "questions": Q_GENERIC, "lang": "en-US"}),
    ("explicit_lang_fr", {}, {"state": {"body": "hello"}, "questions": Q_GENERIC, "lang": "fr"}),
    ("td_workflow_auto_off", {}, {"state": {"body": "I was charged twice"}, "questions": _td_questions(_TD_IDS["customer_service"])}),
    ("empty_state", {}, {"state": {}, "questions": Q_GENERIC}),
    ("none_state", {}, {"state": None, "questions": Q_GENERIC}),
    ("no_questions_none", {}, {"state": {"body": "मुझसे दो बार"}, "questions": None}),

    # ---- auto_task_detection opt-in, all four workflow signatures ----
    ("td_agent_trace_observability_auto_on", {"auto_task_detection": True},
     {"state": {"body": "trace"}, "questions": _td_questions(_TD_IDS["agent_trace_observability"])}),
    ("td_customer_service_auto_on", {"auto_task_detection": True},
     {"state": {"body": "I was charged twice"}, "questions": _td_questions(_TD_IDS["customer_service"])}),
    ("td_invoice_processing_auto_on", {"auto_task_detection": True},
     {"state": {"body": "invoice mismatch"}, "questions": _td_questions(_TD_IDS["invoice_processing"])}),
    ("td_security_incidents_auto_on", {"auto_task_detection": True},
     {"state": {"body": "possible breach"}, "questions": _td_questions(_TD_IDS["security_incidents"])}),
    ("td_auto_on_but_generic_questions", {"auto_task_detection": True},
     {"state": {"body": "I was charged twice"}, "questions": Q_GENERIC}),
    ("td_explicit_beats_workflow", {"auto_task_detection": True},
     {"state": {"body": "x"}, "questions": _td_questions(_TD_IDS["customer_service"]), "model": "multilingual"}),
    # near-miss id sets: must NOT match any workflow even with auto_task_detection on
    ("td_near_miss_partial_overlap", {"auto_task_detection": True},
     {"state": {"body": "x"}, "questions": {"urgency": {"type": "noul", "instructions": "x"},
                                             "category": {"type": "noul", "instructions": "x"}}}),
    ("td_near_miss_superset", {"auto_task_detection": True},
     {"state": {"body": "x"}, "questions": _td_questions(_TD_IDS["customer_service"] + ["extra_field"])}),
    ("td_near_miss_subset", {"auto_task_detection": True},
     {"state": {"body": "x"}, "questions": _td_questions(_TD_IDS["invoice_processing"][:-1])}),
    ("td_near_miss_one_id_swapped", {"auto_task_detection": True},
     {"state": {"body": "x"},
      "questions": _td_questions(_TD_IDS["security_incidents"][:-1] + ["not_a_real_field"])}),

    # ---- unknown-Latin routing (#35): scripts with no stopword list, on the diacritic signal ----
    ("unknown_latin_romanian", {}, {"state": "Gătește-mi o rețetă de sarmale de post pentru mâine."}),
    ("unknown_latin_romanian_agent_request", {}, {"state": "Exportă APK-ul pentru Android și pune-l pe Drive ca să-l instalez."}),
    ("unknown_latin_polish", {}, {"state": "Klient został obciążony dwukrotnie i chce zwrot pieniędzy za fakturę"}),
    ("unknown_latin_turkish", {}, {"state": "Müşteriden iki kez ücret alındı ve para iadesi istiyor lütfen yardım"}),
    ("unknown_latin_undecided_reason", {}, {"state": "Müşteriden iki kez ücret alındı ve para iadesi istiyor"}),
    ("english_still_english", {}, {"state": "Please refund the duplicate charge on invoice 4411 today."}),
    ("short_english_still_english", {}, {"state": "refund me"}),

    # ---- plain-ASCII Romance / accented forms (#172) ----
    ("plain_ascii_es_1", {}, {"state": "El pedido llego roto y nadie responde cuando escribo al soporte"}),
    ("plain_ascii_it_1", {}, {"state": "Il cliente e stato addebitato due volte e vuole un rimborso"}),
    ("plain_ascii_pt_1", {}, {"state": "O cliente foi cobrado duas vezes e quer o dinheiro de volta"}),
    ("plain_ascii_fr_1", {}, {"state": "Le client a ete facture deux fois et demande un remboursement"}),
    ("accented_es", {}, {"state": "La facturación tiene un error y necesito una corrección urgente"}),
    ("accented_it", {}, {"state": "La fattura è sbagliata, devo avere un rimborso per il pagamento"}),
    ("accented_fr", {}, {"state": "La commande est arrivée cassée et personne ne répond au support"}),
    ("romance_control_1", {}, {"state": "The customer was charged twice and wants a refund for this invoice"}),
    ("romance_control_2", {}, {"state": "Our MI5 and UN contacts discussed the DOS attack in LA last month"}),
    ("shared_words_still_multilingual", {}, {"state": "Cât e ora acum la Tokyo"}),

    # ---- custom default ----
    ("custom_default_multilingual", {"default": "multilingual"}, {"state": "12345"}),
    ("custom_default_typed_decisions", {"default": "typed-decisions"}, {"state": ""}),
    ("custom_default_english_script_unknown", {"default": "english"}, {"state": "12345 6789"}),

    # ---- bundle vs standalone repos, local override ----
    ("bundle_default_hindi", {}, {"state": {"m": "मुझसे दो बार"}, "questions": Q_GENERIC}),
    ("standalone_opt_in_hindi", {"standalone_repos": True}, {"state": {"m": "मुझसे दो बार"}, "questions": Q_GENERIC}),
    ("standalone_english_unchanged", {"standalone_repos": True}, {"state": {"m": "I was charged twice"}, "questions": Q_GENERIC}),
    ("local_path_override", {"models": {"english": "/tmp/en", "multilingual": "/tmp/ml"}},
     {"state": {"m": "मुझसे दो बार"}, "questions": Q_GENERIC}),

    # ---- every alias, as an explicit `model=` (normalise_name feeds straight into RouteDecision) ----
    ("alias_en", {}, {"state": "x", "model": "en"}),
    ("alias_laya", {}, {"state": "x", "model": "laya"}),
    ("alias_default", {}, {"state": "x", "model": "default"}),
    ("alias_multi", {}, {"state": "x", "model": "multi"}),
    ("alias_ml", {}, {"state": "x", "model": "ml"}),
    ("alias_laya_multilingual", {}, {"state": "x", "model": "laya-multilingual"}),
    ("alias_typed", {}, {"state": "x", "model": "typed"}),
    ("alias_typed_decisions", {}, {"state": "x", "model": "typed_decisions"}),
    ("alias_laya_typed_decisions", {}, {"state": "x", "model": "laya-typed-decisions"}),
    ("alias_decisions", {}, {"state": "x", "model": "decisions"}),
    ("alias_uppercase_ML", {}, {"state": "x", "model": "ML"}),
    ("alias_mixed_case_English", {}, {"state": "x", "model": "English"}),
    ("alias_via_task_typed_decisions_with_dash", {}, {"state": "x", "task": "typed-decisions"}),
    ("alias_via_task_typed_decisions_with_underscore", {}, {"state": "x", "task": "typed_decisions"}),

    # ---- decision payload shape (repo string, reason text, detection dict) ----
    ("payload_shape_hindi", {}, {"state": {"body": "मुझसे दो बार शुल्क लिया गया"}, "questions": Q_GENERIC}),
    ("payload_shape_armenian_non_latin_percent", {}, {"state": "Հայերեն abc"}),
    ("payload_shape_unidentified_latin_percent", {}, {"state": "Müşteriden iki kez ücret alındı ve para iadesi istiyor"}),
]


# ============================================================================ email_probe

_DISCLAIMER_TEXT = "This email is confidential and intended solely for the named addressee."

EMAIL_CLEAN_CASES = [
    ("upstream/confidential_request", "Confidential: I need a refund.", 3000),
    ("upstream/from_prose", "Hello.\nFrom: my side the integration works, but please refund it.", 3000),
    ("upstream/from_header", "Please refund it.\nFrom: Jane Smith\nSent: Monday\nOld request.", 3000),
    ("upstream/thanks_prose", "Hello.\nThanks for the quick reply, but I need a refund.", 3000),
    ("upstream/english_signoff", "Please refund it.\nWarmest regards, Łukasz", 3000),
    ("upstream/lowercase_name_is_prose", "Please refund it.\nThanks, żaneta needs help.", 3000),
    ("upstream/marked_name", "Please refund it.\nRegards, Jose\u0301", 3000),
    ("upstream/astral_signoff_name", "Please refund it.\nRegards, \U00010400\U00010428", 3000),
    ("upstream/unattached_mark", "Please refund it.\nRegards, \u0301Jose", 3000),
    ("upstream/portuguese_quote", "Preciso de ajuda.\nEm 10/09 João escreveu:\nOld request.", 3000),
    ("upstream/spanish_quote", "Necesito ayuda.\nEl 10/09 Juan escribió:\nOld request.", 3000),
    ("upstream/french_quote", "Je demande un remboursement.\nLe 10/09 Jean a écrit :\nOld request.", 3000),
    ("upstream/wrapped_attribution",
     "Please refund it.\nOn 10/09 Jane Smith\njane@example.com> wrote:\nOld request.", 3000),
    ("upstream/french_signoff", "Je demande un remboursement.\nCordialement,\nMarie", 3000),
    ("upstream/portuguese_signoff", "Preciso de ajuda.\nAtenciosamente,\nMaria", 3000),
    ("upstream/spanish_signoff", "Necesito ayuda.\nSaludos cordiales,\nJuan", 3000),
    ("upstream/device_footer", "Please refund it.\nSent from my Samsung Galaxy.", 3000),
    ("upstream/device_mention_prose", "Please refund it.\nSent from my iPhone yesterday to support.", 3000),
    ("upstream/french_disclaimer",
     "Je demande un remboursement.\n\nCe message est confidentiel et réservé au destinataire.", 3000),
    ("upstream/french_device_footer", "Merci de rembourser la facture.\n\nEnvoyé depuis mon iPhone", 3000),
    ("upstream/french_device_footer_de_mon", "Merci de rembourser la facture.\n\nEnvoyé de mon iPad.", 3000),
    ("upstream/french_device_footer_no_possessive", "Merci de rembourser la facture.\n\nEnvoyé depuis Outlook", 3000),
    ("upstream/french_device_mention_prose",
     "Bonjour,\nEnvoyé depuis mon iPhone par erreur.\nMerci de rembourser la facture.", 3000),
    # ---- straight from tests/test_email.py ----
    ("inline_footer_no_blank_line", "My account is locked.\n%s\nPlease unlock it." % _DISCLAIMER_TEXT, 3000),
    ("inline_footer_no_terminal_punctuation", "My account is locked\n%s\nPlease unlock it." % _DISCLAIMER_TEXT, 3000),
    ("inline_footer_body_never_emptied", "My account is locked. %s" % _DISCLAIMER_TEXT, 3000),
    ("standalone_footer_paragraph_dropped", "My account is locked.\n\n%s" % _DISCLAIMER_TEXT, 3000),
    ("wrapped_standalone_footer_dropped",
     "My account is locked.\n\nThis email and any files transmitted with it are\n"
     "confidential and intended solely for the named addressee.", 3000),
    ("received_in_error_footer_dropped",
     "Please reopen ticket 4411.\n\nIf you have received this message in error, delete it.", 3000),
    ("quoted_history_removed", "Thanks for the update.\nOn Mon, Sep 20, Bob wrote:\n> original text", 3000),
    ("signature_block_removed", "Hi team,\nCan you confirm the refund?\nRegards,\nAlice", 3000),
    ("empty_body", "", 3000),

    # ---- quote header variants (each _QUOTE_HEADERS pattern) ----
    ("quote_header_on_wrote_long_name",
     "Please see below.\nOn Mon, Jan 5, 2026 at 3:00 PM, Bob Smith <bob@example.com> wrote:\n> old text here", 3000),
    ("quote_header_original_message",
     "Approved.\n----- Original Message -----\nFrom: alice@example.com\nSubject: Re: Invoice", 3000),
    ("quote_header_forwarded_message",
     "See below for context.\n---------- Forwarded Message ----------\nFrom: bob@example.com", 3000),
    ("quote_header_underscores",
     "My reply is above.\n________________________________\nFrom: system@example.com", 3000),
    ("quote_header_from_line",
     "Confirmed, thanks.\nFrom: support@example.com\nSubject: Ticket 4411 update\nDate: today", 3000),
    ("quote_header_only_no_body_before_kept",
     "On Mon, Jan 5, 2026, Bob wrote:\n> quoted only, no lines before it to break on", 3000),
    ("quote_marker_gt_lines_stripped",
     "My reply.\n> quoted line one\n> quoted line two\nMore of my reply.", 3000),

    # ---- signature marker variants, at various positions ----
    ("signature_dash_dash", "Hi,\nPlease confirm.\n--\nAlice Smith\nSupport Team", 3000),
    ("signature_best_regards", "Hi,\nCan you confirm the refund?\nBest regards,\nAlice", 3000),
    ("signature_kind_regards", "Hi,\nCan you confirm the refund?\nKind regards,\nBob", 3000),
    ("signature_warm_regards", "Hi,\nCan you confirm the refund?\nWarm regards,\nCarol", 3000),
    ("signature_many_thanks", "Hi,\nCan you confirm the refund?\nMany thanks,\nDave", 3000),
    ("signature_thanks", "Hi,\nCan you confirm the refund?\nThanks,\nEve", 3000),
    ("signature_thank_you", "Hi,\nCan you confirm the refund?\nThank you,\nFrank", 3000),
    ("signature_cheers", "Hi,\nCan you confirm the refund?\nCheers,\nGrace", 3000),
    ("signature_sincerely", "Hi,\nCan you confirm the refund?\nSincerely,\nHeidi", 3000),
    ("signature_sent_from_iphone", "On the go, will follow up later.\nSent from my iPhone", 3000),
    ("signature_sent_from_android", "Quick reply here.\nSent from my Android", 3000),
    ("signature_too_early_kept_as_body",
     # A "Thanks," line inside the first 60% of a short body is not treated as a signature
     # cutoff by the search window (range starts at max(1, int(len(lines)*0.6), len(lines)-8)));
     # this pins that boundary behaviour rather than assuming it.
     "Thanks,\nfor looking into this.\nMy account is still locked after the reset.\n"
     "Please escalate this to the on-call engineer today.\nRegards,\nIvan", 3000),

    # ---- disclaimers inside and across paragraphs ----
    ("disclaimer_mid_paragraph_request_survives",
     "My account is locked.\n%s\nPlease unlock it." % _DISCLAIMER_TEXT, 3000),
    ("disclaimer_two_sentences_only_boilerplate_dropped",
     "My account is locked. %s Please help urgently." % _DISCLAIMER_TEXT, 3000),
    ("disclaimer_across_two_paragraphs_only_its_own_paragraph_affected",
     "Please reopen ticket 4411 as soon as you can.\n\n"
     "This email and any attachments are confidential and intended solely for the addressee.\n\n"
     "If you have received this message in error, please delete it.\n\n"
     "Let me know once it is done.", 3000),
    ("disclaimer_received_in_error_variant_2",
     "Approve the refund please.\n\nIf you received this e-mail in error, notify the sender immediately.", 3000),

    # ---- CRLF and literal backslash-n ----
    ("crlf_line_endings", "My account is locked.\r\nPlease unlock it.\r\nRegards,\r\nAlice", 3000),
    ("crlf_with_quote_header", "Thanks for the update.\r\nOn Mon, Sep 20, Bob wrote:\r\n> original text", 3000),
    ("literal_backslash_n_becomes_newline",
     "My account is locked.\\nPlease unlock it.\\nRegards,\\nAlice", 3000),
    ("literal_backslash_n_with_disclaimer",
     "My account is locked.\\n%s\\nPlease unlock it." % _DISCLAIMER_TEXT, 3000),
    ("mixed_real_and_literal_newlines",
     "Line one.\nLine two.\\nLine three (literal escape).\nLine four.", 3000),

    # ---- Unicode whitespace: `[ \\t]+` in clean_email_body only collapses ASCII space/tab, not
    #      general Unicode whitespace; `.strip()` on paragraphs strips all Unicode whitespace ----
    ("unicode_nbsp_not_collapsed",
     "My account is locked.\nPlease unlock it.", 3000),
    ("unicode_em_space_and_ideographic_space",
     "My account is　locked.\nPlease unlock it.", 3000),
    ("unicode_whitespace_leading_trailing_stripped",
     "  My account is locked. \n Please unlock it.  ", 3000),
    ("unicode_zero_width_space_kept",
     "My​account is locked.\nPlease​unlock it.", 3000),

    # ---- max_chars variants (default 3000, and non-default) ----
    ("max_chars_tiny_cuts_mid_word", "Please refund the duplicate charge on invoice 4411 today.", 10),
    ("max_chars_default_no_truncation_needed", "Short message that fits easily.", 3000),
    ("max_chars_large_no_op", "Please refund the duplicate charge on invoice 4411 today.", 100000),
    ("max_chars_exact_boundary",
     "0123456789" * 5, 50),  # exactly 50 chars input, max_chars=50 -> no truncation
    ("max_chars_one_over_boundary",
     "0123456789" * 5 + "X", 50),  # 51 chars, max_chars=50 -> drop last char

    # ---- long, multi-paragraph, everything at once (kitchen sink) ----
    ("kitchen_sink_quote_signature_disclaimer",
     "Hi team,\n\n"
     "My account was charged twice for the Pro plan this month and the refund has not "
     "arrived yet. Could you please look into this today?\n\n"
     "Best regards,\n"
     "Priya Sharma\n"
     "Sent from my iPhone\n\n"
     "This email and any files transmitted with it are confidential and intended solely "
     "for the use of the individual or entity to whom they are addressed. If you have "
     "received this email in error please notify the sender.\n\n"
     "On Tue, Sep 22, 2026 at 9:14 AM, Support <support@example.com> wrote:\n"
     "> Hi Priya, thanks for reaching out, we are looking into it.\n"
     "> - Support Team\n", 3000),
]

EMAIL_STATE_CASES = [
    # (label, subject, body, sender, clean, extra)
    ("basic_clean_true", "Locked out", "My account is locked. %s" % _DISCLAIMER_TEXT, None, True, {}),
    ("basic_clean_false_keeps_raw_body", "Locked out", "My account is locked.\n%s" % _DISCLAIMER_TEXT, None, False, {}),
    ("with_sender", "Refund request", "Please refund invoice 4411.", "user@example.com", True, {}),
    ("subject_whitespace_stripped", "   Refund request   ", "Please help.", None, True, {}),
    ("extra_fields_kept_when_not_none", "Ticket", "Body text.", "user@example.com", True, {"priority": "high", "ticket_id": 4411}),
    ("extra_none_values_dropped", "Ticket", "Body text.", None, True, {"nullable": None, "kept": "value"}),
    ("empty_subject_and_body", "", "", None, True, {}),
    ("hindi_subject_and_body", "सहायता चाहिए", "मुझसे दो बार शुल्क लिया गया, कृपया रिफंड करें।", None, True, {}),
]


# ============================================================================ shortlist_probe
#
# Each entry supplies a positional embedding matrix directly (query vector first, then one
# vector per criteria item in order) rather than a text-keyed table, so the golden dump needs
# no embedder at all: `embed_fn` for these cases is just "return this matrix, in this order",
# and the dump script asserts it is called with exactly `len(vectors)` texts when `vectors`
# is not None. A case with `vectors=None` asserts k >= n passthrough: embed_fn must not be
# called at all (checked by the dump script raising if it is).

SHORTLIST_CASES = [
    # ---- deterministic top-k, ties keep the earlier label ----
    {
        "label": "tie_cosine_keeps_earlier_label",
        "state": "pay me", "criteria": {"alpha": None, "beta": "", "gamma": "mid", "delta": "same"},
        "k": 2, "instructions": None,
        "vectors": [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.6, 0.8], [1.0, 0.0]],
    },
    {
        "label": "k1_is_earliest_max",
        "state": "pay me", "criteria": {"alpha": None, "beta": "", "gamma": "mid", "delta": "same"},
        "k": 1, "instructions": None,
        "vectors": [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.6, 0.8], [1.0, 0.0]],
    },
    {
        "label": "k3_appends_next_cosine",
        "state": "pay me", "criteria": {"alpha": None, "beta": "", "gamma": "mid", "delta": "same"},
        "k": 3, "instructions": None,
        "vectors": [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.6, 0.8], [1.0, 0.0]],
    },
    # ---- zero query vector: every cosine is 0, earliest labels win ----
    {
        "label": "zero_query_keeps_original_order",
        "state": "pay me", "criteria": {"alpha": None, "beta": "", "gamma": "mid", "delta": "same"},
        "k": 2, "instructions": None,
        "vectors": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.6, 0.8], [3.0, 4.0]],
    },
    # ---- zero option vector loses to a real match ----
    {
        "label": "zero_option_vector_loses",
        "state": "pay me", "criteria": {"alpha": None, "beta": None, "gamma": None},
        "k": 2, "instructions": None,
        "vectors": [[1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    },
    # ---- NaN / inf: non-finite embeddings become 0 (np.nan_to_num), so they never outrank a
    #      finite match, but inf-vs-inf or nan-vs-nan still tie at 0 and fall back to order ----
    {
        "label": "nan_vector_sorts_behind_finite_match",
        "state": "pay me", "criteria": {"alpha": None, "beta": None},
        "k": 1, "instructions": None,
        "vectors": [[1.0, 0.0], [float("nan"), float("nan")], [1.0, 0.0]],
    },
    {
        "label": "positive_inf_and_negative_inf_both_zeroed",
        "state": "pay me", "criteria": {"alpha": None, "beta": None, "gamma": None},
        "k": 2, "instructions": None,
        "vectors": [[1.0, 0.0], [float("inf"), float("inf")], [float("-inf"), 0.0], [1.0, 0.0]],
    },
    {
        # nan_to_num zeroes only the NaN component, not the whole vector: [nan, 1.0] becomes
        # [0.0, 1.0], a real unit vector, so this is NOT the same as a zero query -- it still
        # picks a winner (beta) rather than falling back to insertion order.
        "label": "nan_component_in_query_only_zeroes_that_component",
        "state": "pay me", "criteria": {"alpha": None, "beta": None},
        "k": 1, "instructions": None,
        "vectors": [[float("nan"), 1.0], [1.0, 0.0], [0.0, 1.0]],
    },
    {
        # Every component of the query is NaN, so nan_to_num zeroes the whole vector: this IS
        # a true zero query, and falls back to insertion order like zero_query_keeps_original_order.
        "label": "fully_nan_query_falls_back_to_order",
        "state": "pay me", "criteria": {"alpha": None, "beta": None},
        "k": 1, "instructions": None,
        "vectors": [[float("nan"), float("nan")], [1.0, 0.0], [0.0, 1.0]],
    },
    # ---- k >= n passthrough: embed_fn must not be called ----
    {
        "label": "k_equals_n_passthrough",
        "state": "pay me", "criteria": {"alpha": None, "beta": "", "gamma": "mid", "delta": "same"},
        "k": 4, "instructions": None, "vectors": None,
    },
    {
        "label": "k_greater_than_n_passthrough",
        "state": "pay me", "criteria": {"alpha": None, "beta": "", "gamma": "mid", "delta": "same"},
        "k": 20, "instructions": None, "vectors": None,
    },
    {
        "label": "k_equals_n_single_option_passthrough",
        "state": "pay me", "criteria": {"solo": "the only choice"},
        "k": 1, "instructions": None, "vectors": None,
    },
    # ---- list criteria (positional labels, value=None each) ----
    {
        "label": "list_criteria_topk",
        "state": "pay me", "criteria": ["alpha", "beta", "gamma"],
        "k": 2, "instructions": "Classify",
        "vectors": [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.2]],
    },
    {
        "label": "list_criteria_passthrough",
        "state": "hello", "criteria": ["alpha", "beta", "gamma"],
        "k": 5, "instructions": "Which?", "vectors": None,
    },
    # ---- dict state serialisation through query text (informational only: matrix is positional) ----
    {
        "label": "dict_state_query",
        "state": {"text": "hi"}, "criteria": ["alpha", "beta"],
        "k": 1, "instructions": "Classify",
        "vectors": [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    },
    # ---- 0 and False are real criterion values (not "no description") ----
    {
        "label": "zero_and_false_criteria_values",
        "state": "pay me", "criteria": {"zero": 0, "no": False, "bare": None, "named": {"desc": "payments"}},
        "k": 4, "instructions": None, "vectors": None,   # k == n -> passthrough, but exercises render text
    },
    {
        "label": "zero_and_false_criteria_ranked",
        "state": "pay me", "criteria": {"zero": 0, "no": False, "bare": None, "named": {"desc": "payments"}},
        "k": 2, "instructions": None,
        "vectors": [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [0.0, 1.0]],
    },
    # ---- non-string instructions: dict / list / number, run through json.dumps(ensure_ascii=False) ----
    {
        "label": "dict_instructions",
        "state": "pay me", "criteria": {"alpha": None, "beta": None},
        "k": 1, "instructions": {"task": "route", "hint": "Müller & co <urgent>"},
        "vectors": [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    },
    {
        "label": "list_instructions",
        "state": "pay me", "criteria": {"alpha": None, "beta": None},
        "k": 1, "instructions": ["step one", "step two", 3],
        "vectors": [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
    },
    {
        "label": "numeric_instructions",
        "state": "pay me", "criteria": {"alpha": None, "beta": None},
        "k": 1, "instructions": 42,
        "vectors": [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    },
    {
        "label": "empty_string_instructions_same_as_none",
        "state": "pay me", "criteria": {"alpha": None, "beta": None},
        "k": 1, "instructions": "",
        "vectors": [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    },
    # ---- many options, mid-range k (mirrors the checkpoint end-to-end case's shape at n=40) ----
    {
        "label": "many_options_k5_of_10",
        "state": "I need help exporting invoices to a spreadsheet",
        "criteria": {"topic_%02d" % i: "detailed description number %d for this support topic" % i for i in range(10)},
        "k": 5, "instructions": "Pick the single best matching category.",
        "vectors": [[1.0, 0.0]] + [[float(i % 3), float((i + 1) % 3)] for i in range(10)],
    },
]


# ============================================================================ sample inputs
#
# Single source of truth for the combined routing sample (Laya.Routing / laya-routing), so
# both languages' samples read the same fixed content and their stdout can be diffed.

# Section 1: ~8 detection states (English, accent-stripped Spanish, French, Romanian, Hindi,
# Arabic, Japanese, a dict state).
SAMPLE_DETECTION_STATES = [
    ("english", "The customer was charged twice this month and wants a refund."),
    ("spanish_accent_stripped", "El cliente fue cobrado dos veces y quiere que le devuelvan el dinero"),
    ("french", "Le client a été facturé deux fois et demande un remboursement."),
    ("romanian", "Am fost taxat de două ori pentru factura din luna martie și vreau banii"),
    ("hindi", "ग्राहक से दो बार शुल्क लिया गया और वह धनवापसी चाहता है।"),
    ("arabic", "تم خصم المبلغ مرتين من العميل ويريد استرداد الأموال"),
    ("japanese", "お客様は二重に請求されたため返金を希望しています。"),
    ("dict_state", {"subject": "Double charge", "body": "मुझसे दो बार शुल्क लिया गया"}),
]

# Section 2: routing-decision examples, including explicit model/lang and opt-in workflow
# detection, with no engine loaded (Route only).
SAMPLE_ROUTING_DECISIONS = [
    ("auto_english", {}, {"state": "Please refund the duplicate charge on invoice 4411 today."}),
    ("auto_hindi", {}, {"state": "मुझसे दो बार शुल्क लिया गया, कृपया रिफंड करें।"}),
    ("auto_unknown_latin_romanian", {}, {"state": "Gătește-mi o rețetă de sarmale de post pentru mâine."}),
    ("explicit_model_multilingual", {}, {"state": "anything at all", "model": "multilingual"}),
    ("explicit_lang_de", {}, {"state": "hello there", "lang": "de"}),
    ("workflow_opt_in_customer_service", {"auto_task_detection": True},
     {"state": {"body": "I was charged twice"},
      "questions": _td_questions(_TD_IDS["customer_service"])}),
    ("workflow_opt_in_off_by_default", {},
     {"state": {"body": "I was charged twice"},
      "questions": _td_questions(_TD_IDS["customer_service"])}),
]

# Section 3: an English and a Hindi support email, same support questions, run through
# Router.predict (engine loaded, MaxLoaded=2).
SAMPLE_SUPPORT_QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "sales": "pricing, new contracts",
            "other": "everything else",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"],
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel or leave?",
    },
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the user explicitly request a refund?",
    },
}

SAMPLE_SUPPORT_EMAIL_EN = {
    "from": "user@acme.com",
    "subject": "Duplicate charge on invoice #4411",
    "body": "Hi, we were billed twice for March. Please refund the duplicate today "
            "or we will cancel our plan.",
}

# Written as escapes deliberately (see laya-dotnet/tools/sample_cases.py's note on the same pitfall): a
# hand-retyped Devanagari string is an easy place to silently swap one similar-looking
# character for another and produce a "parity bug" that is really a transcription error.
SAMPLE_SUPPORT_EMAIL_HI = {
    "from": "user@acme.com",
    "subject": "चालान में दोहरा "
               "शुल्क",
    "body": "मार्च का बिल "
            "दो बार लिया गया, "
            "कृपया रिफंड करें।",
}

# Section 4: a raw email with quoted history, signature and disclaimer, run through
# LayaEmail.CleanBody -> LayaEmail.State -> LayaPresets.Email() through the router.
SAMPLE_RAW_EMAIL_SUBJECT = "Re: Refund for invoice 4411"
SAMPLE_RAW_EMAIL_SENDER = "priya.sharma@example.com"
SAMPLE_RAW_EMAIL_BODY = (
    "Hi team,\n\n"
    "I was charged twice for the Pro plan this month and the second charge still has "
    "not been refunded. This is the third time I have written about it -- please "
    "resolve this today or I will need to cancel the account.\n\n"
    "Best regards,\n"
    "Priya Sharma\n"
    "Sent from my iPhone\n\n"
    "This email and any files transmitted with it are confidential and intended "
    "solely for the use of the individual or entity to whom they are addressed. If "
    "you have received this email in error please notify the sender.\n\n"
    "On Tue, Sep 22, 2026 at 9:14 AM, Support <support@example.com> wrote:\n"
    "> Hi Priya, thanks for reaching out, we are looking into it.\n"
    "> - Support Team\n"
)

# Section 5: a 40-intent banking choice question through LayaShortlist.Predict with the
# hashing embedder (demo only).
SAMPLE_BANKING_STATE = "I tried to withdraw cash from the ATM but it kept the card and gave me no money."
SAMPLE_BANKING_INSTRUCTIONS = "What is the customer's banking intent?"
SAMPLE_BANKING_CRITERIA = {
    "activate_my_card": "turn on a newly received card",
    "age_limit": "minimum or maximum age to use the service",
    "apple_pay_or_google_pay": "adding the card to a mobile wallet",
    "atm_support": "which ATMs the card works at",
    "automatic_top_up": "automatically topping up the balance",
    "balance_not_updated_after_bank_transfer": "a bank transfer is missing from the balance",
    "balance_not_updated_after_cheque_or_cash_deposit": "a deposit is missing from the balance",
    "beneficiary_not_allowed": "cannot add a payment recipient",
    "cancel_transfer": "wants to cancel a transfer already sent",
    "card_about_to_expire": "the card is close to its expiry date",
    "card_acceptance": "where the card is accepted",
    "card_arrival": "asking when a new card will arrive",
    "card_delivery_estimate": "asking how long delivery will take",
    "card_linking": "linking a card to the account",
    "card_not_working": "the card is declined or not working",
    "card_payment_fee_charged": "an unexpected fee was charged on a card payment",
    "card_payment_not_recognised": "a card payment the customer does not recognise",
    "card_payment_wrong_exchange_rate": "the exchange rate used was wrong",
    "card_swallowed": "the ATM kept the card",
    "cash_withdrawal_charge": "an unexpected charge for a cash withdrawal",
    "cash_withdrawal_not_recognised": "a cash withdrawal the customer does not recognise",
    "change_pin": "wants to change the PIN",
    "compromised_card": "the card may have been compromised",
    "contactless_not_working": "contactless payment is not working",
    "country_support": "which countries the service supports",
    "declined_card_payment": "a card payment was declined",
    "declined_cash_withdrawal": "a cash withdrawal was declined",
    "declined_transfer": "a transfer was declined",
    "direct_debit_payment_not_recognised": "a direct debit the customer does not recognise",
    "disposable_card_limits": "limits on a disposable virtual card",
    "edit_personal_details": "wants to edit name, address or other personal details",
    "exchange_charge": "a fee charged for currency exchange",
    "exchange_rate": "asking what the current exchange rate is",
    "exchange_via_app": "exchanging currency inside the app",
    "extra_charge_on_statement": "an unexplained extra charge on the statement",
    "failed_transfer": "a transfer failed",
    "fiat_currency_support": "which regular currencies are supported",
    "get_disposable_virtual_card": "wants a disposable virtual card",
    "get_physical_card": "wants a physical card",
    "getting_spare_card": "wants a spare or backup card",
}
