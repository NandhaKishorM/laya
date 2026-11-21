package router

import (
	"context"

	"github.com/NandhaKishorM/laya/laya-go/agent"
	"github.com/NandhaKishorM/laya/laya-go/schema"
)

// Runner lets a test stand in for the agent's inference and see exactly what the Router asked.
type Runner struct {
	Predict func(ctx context.Context, a *agent.Agent, state any, qs schema.Questions, ac AgentCall) (schema.Prediction, error)
	Batch   func(ctx context.Context, a *agent.Agent, states []any, qs schema.Questions, ac AgentCall) ([]schema.Prediction, error)
	Long    func(ctx context.Context, a *agent.Agent, state any, qs schema.Questions, ac AgentCall) (agent.LongPrediction, error)
}

func (x Runner) predict(ctx context.Context, a *agent.Agent, state any, qs schema.Questions, ac AgentCall) (schema.Prediction, error) {
	return x.Predict(ctx, a, state, qs, ac)
}

func (x Runner) predictBatch(ctx context.Context, a *agent.Agent, states []any, qs schema.Questions, ac AgentCall) ([]schema.Prediction, error) {
	return x.Batch(ctx, a, states, qs, ac)
}

func (x Runner) predictLong(ctx context.Context, a *agent.Agent, state any, qs schema.Questions, ac AgentCall) (agent.LongPrediction, error) {
	return x.Long(ctx, a, state, qs, ac)
}

// SetRunner replaces the Router's inference with x.
func SetRunner(r *Router, x Runner) { r.rn = x }
