package email

import "regexp"

// In the view a space is one of Python's 29 space characters (the non-ASCII ones are rewritten
// to a plain space) and a digit is [0-9].
const (
	sp = `[\t-\r\x1c-\x20\x0e]`
	ns = `[^\t-\r\x1c-\x20\x0e]`
	dg = `[0-9\x0f]`
	// spIn is the space class without brackets, for adding members.
	spIn = `\t-\r\x1c-\x20\x0e`
)

func compile(pattern string) *regexp.Regexp { return regexp.MustCompile(translate(pattern)) }

// quoteHeader is one pattern of _QUOTE_HEADERS. needsDigit is the lookahead "(?=.*\d)" the
// reference puts after the leading word: a client's attribution always carries a date, so
// "Em resposta ao que você escreveu:" is body text. The words before the lookahead hold no
// digit, so a digit anywhere in the line is the same test.
type quoteHeader struct {
	re         *regexp.Regexp
	needsDigit bool
}

func (q quoteHeader) matches(view string) bool {
	return q.re.MatchString(view) && (!q.needsDigit || hasDigit(view))
}

var quoteHeaders = []quoteHeader{
	{compile(`^` + sp + `*on .{0,300}wrote:` + sp + `*$`), false},
	{compile(`^` + sp + `*em .{0,300}escreveu:` + sp + `*$`), true},
	{compile(`^` + sp + `*el .{0,300}escribi[óo]:` + sp + `*$`), true},
	{compile(`^` + sp + `*le .{0,300}a [eé]crit` + sp + `*:` + sp + `*$`), true},
	{compile(`^` + sp + `*-{2,}` + sp + `*(original|forwarded) message` + sp + `*-{2,}`), false},
	{compile(`^` + sp + `*-{2,}` + sp + `*(mensagem (original|encaminhada)|mensaje (original|reenviado)|message d'origine)` +
		sp + `*-{2,}`), false},
	{compile(`^` + sp + `*_{8,}` + sp + `*$`), false},
	// `From:` and `De:` open ordinary prose too, so the header needs an address after it.
	{compile(`^` + sp + `*from:` + sp + `.*[@<]`), false},
	{compile(`^` + sp + `*de` + sp + `*:` + sp + `.*[@<]`), false},
}

// attributionTail is Gmail's wrapped attribution, `fulano@x.com> escreveu:` alone on a line. It
// takes the `On/Em/El/Le ...` head it belongs to with it.
var (
	attributionTail = compile(`^.{0,120}` + ns + `@` + ns + `+` + sp + `+(wrote|escreveu|escribi[óo]|a [eé]crit)` +
		sp + `*:` + sp + `*$`)
	attributionHead = compile(`^` + sp + `*(on|em|el|le) `) // plus a digit later in the line
)

// A bare `De: Maria Souza` or `From: Maria Souza` only cuts when the header's own `Sent:` or a
// dated `Data:` line follows it.
var (
	headerFromName = compile(`^` + sp + `*(de|from)` + sp + `*:` + sp + `+` + ns)
	headerNext     = compile(`^` + sp + `*(enviad[oa]( em| el)?:` + sp + `|envoy[ée]( le)?` + sp + `*:` + sp +
		`|sent:` + sp + `|(data|fecha|date):` + sp + `.*` + dg + `{4})`)
)

// The English sign-off is the head here plus a name judged by isEnglishSignoff; the case of the
// name matters and re.I would erase it, so only the closing words are matched on the view.
var signoffHead = compile(`^` + sp + `*(best|kind|warmest|warm|many thanks|thanks|thank you|regards|cheers|sincerely)` +
	`(` + sp + `+(and|&)` + sp + `+regards|` + sp + `+(regards|wishes|again|in advance|a lot|so much|very much))?`)

var (
	signoffDashes = compile(`^` + sp + `*--` + sp + `*$`)

	// Portuguese, Spanish and French sign-offs match only on their own: "Obrigado pelo retorno,
	// mas ..." is a request, not a signature.
	signoffRomance = compile(`^` + sp + `*(atenciosamente|att|abraços?|abs|um abraço|cordialmente|grat[oa]|` +
		`(muito )?obrigad[oa]s?( desde já| pela atenção)?|(com os melhores )?cumprimentos|saudações|` +
		`(un )?saludos?( cordiales)?|atentamente|(muchas )?gracias( de antemano)?|` +
		`(bien )?cordialement|salutations( distinguées)?|bien à vous|merci( d'avance)?|` +
		`bonne journée)[` + spIn + `,!.]*$`)
)

const device = `iphone|ipad|android|ios|mobile|celular|telemóvel|móvil|galaxy|smartphone|samsung|tablet|` +
	`outlook|yahoo|mail|e-?mail|gmail|windows`

// deviceFooter matches a line that is nothing but a mobile or mail-app footer.
var deviceFooter = compile(`^` + sp + `*((enviad[oa] (do|pelo|pela|via|desde|a partir do)( meu| minha| mi)?|` +
	`sent from( my)?|envoy[ée] (depuis|de)( mon| ma| mes)?)` +
	` (` + device + `)( (` + device + `|para|for|no|na|` + dg + `+|phone|device|pro|max|mini|plus|` +
	`using [a-z][a-z0-9_.+-]*))*|(obter o|get) outlook (para|for) (ios|android))[` + spIn + `.!]*$`)

// disclaimer is the boilerplate footer, each branch tied to a disclaimer noun and tail so that a
// request which merely mentions "confidential" survives. `[^.]` spans newlines on purpose: a
// footer wraps.
var disclaimer = compile(
	`(\b(e-?mail|message|information|communication|transmission|contents?)\b[^.]{0,60}` +
		`\bconfidential\b[^.]{0,60}\b(intended|solely|addressee|recipient|privileged|` +
		`disclos|unauthori[sz]ed)|` +
		`\bconfidential\b[^.]{0,60}\b(and (may|is) (also )?privileged)|` +
		`if you (have )?received this (e-?mail|message) in error|` +
		`\b(esta|este) (mensagem|e-?mail|mensaje|correo)\b[^.]{0,80}(confidencia|sigilos|privilegiad)|` +
		`\b(uso exclusivo|exclusivamente|únicamente|unicamente)\b[^.]{0,30}` +
		`(destinatári|destinatari|pessoa|persona|entidade|entidad)|` +
		`\b(recebeu|recebido|receber) (esta|este) (mensagem|e-?mail)\b[^.]{0,20} por (engano|erro)|` +
		`\b(ha recibido|recibió|recibe) (este|esta) (mensaje|correo)\b[^.]{0,20} por error|` +
		`\bantes de imprimir\b[^.]{0,100}(meio ambiente|medio ambiente|natureza|planeta|realmente necess)|` +
		`\b(meio|medio) ambiente\b[^.]{0,30}antes de imprimir|` +
		`\b(ce|cet|cette) (message|e-?mail|mail|courriel)\b[^.]{0,80}(confidentiel|privil[eé]gi)|` +
		`\bavez re[çc]u (ce|cet|cette) (message|e-?mail|mail)\b[^.]{0,20} par erreur|` +
		`\b(usage exclusif|exclusivement|uniquement)\b[^.]{0,30}destinataire)`)

// blankLine is re.split's `\n\s*\n`.
var blankLine = compile(`\n` + sp + `*\n`)
