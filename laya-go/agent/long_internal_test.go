package agent

import (
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/schema"
)

func TestMergeCollapsedLaterWindowReplaces(t *testing.T) {
	r := func(id string, total int) schema.CollapsedQuestion {
		return schema.CollapsedQuestion{ID: id, CollapsedOptions: schema.CollapsedOptions{Total: total}}
	}
	var all schema.CollapsedQuestions
	for _, c := range []schema.CollapsedQuestion{r("a", 1), r("b", 2), r("a", 3)} {
		all = mergeCollapsed(all, c)
	}
	if len(all) != 2 || all[0].ID != "a" || all[0].Total != 3 || all[1].ID != "b" || all[1].Total != 2 {
		t.Errorf("merged = %+v, want a replaced in place by its later report, b kept", all)
	}
}
