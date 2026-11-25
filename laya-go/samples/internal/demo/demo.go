// Package demo holds what the samples share: the email questions and the example states.
package demo

import (
	"github.com/NandhaKishorM/laya/laya-go/lang"
	"github.com/NandhaKishorM/laya/laya-go/schema"
)

// emailQuestions is laya's email preset in its wire form.
const emailQuestions = `{
  "category": {"type": "choice", "instructions": "Which team should handle the email in ` + "`body`" + `?",
    "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages, integrations",
      "sales": "pricing, demos, new purchases", "security": "phishing, scams, account compromise",
      "hr": "hiring, leave, payroll", "other": "none of the above"}},
  "is_spam": {"type": "noul", "instructions": "Is this email unsolicited spam or bulk marketing?"},
  "urgency": {"type": "score", "instructions": "How urgent is the request in ` + "`body`" + `?",
    "criteria": ["no time pressure", "needs attention soon", "blocking issue or hard deadline"]},
  "needs_reply": {"type": "noul", "instructions": "Does the sender expect a reply?"}
}`

// Questions returns the four email questions.
func Questions() (schema.Questions, error) { return schema.ParseQuestions([]byte(emailQuestions)) }

// State is a support email. A lang.Object, not a map: a Go map is written in sorted key order
// (body before subject), which is not the order a Python dict gives the model.
func State(subject, body string) lang.Object {
	return lang.Object{{Key: "subject", Value: subject}, {Key: "body", Value: body}}
}
