// Package email cleans mail bodies for classification and builds states from messages. It ports
// laya/email.py.
//
// The markers cover English, Portuguese, Spanish and French mail clients: the router sends
// Portuguese, Spanish and French states to the multilingual checkpoint, and an English-only
// cleaner left Gmail's "Em ... escreveu:", Outlook's "-----Mensagem original-----", the
// "Atenciosamente" sign-off and the confidentiality footer in the input, where the quoted history
// (often a different request) weighed on the answer as much as the new message.
//
// CleanBody removes quoted history, signatures, device footers and disclaimers, and keeps the
// first maxChars characters. State wraps it in the state the agent reads; EmailQuestions, the
// preset questions for mail, is not part of this package yet.
//
// # Why the patterns are not plain regexp
//
// Go's regexp is RE2, so the reference's patterns cannot be pasted in. Python's \s, \d, \w and
// re.I are Unicode relations recorded from CPython (internal/unicodedata), and Go's differ, so
// text is first rewritten to a view with one ASCII byte per character: letters folded the way
// re.I folds them, a character the patterns spell as a literal replaced by a private placeholder,
// every other space, digit and word character replaced by one stand-in of its class. The patterns
// run on the view, where Go's \b, \s and \d mean what Python's mean, and the offsets they report
// are offsets into the original characters. The two rules the reference states as lookahead
// ("Em (?=.*\d)") are the pattern plus a digit test on the same line.
package email
