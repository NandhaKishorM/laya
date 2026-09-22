import unittest
from laya.presets import (
    agent_trace_questions,
    invoice_processing_questions,
    security_incident_questions,
)
from laya.router import match_typed_decisions_workflow

class PresetsWorkflowTests(unittest.TestCase):
    def test_agent_trace_signature_match(self):
        q = agent_trace_questions()
        wf = match_typed_decisions_workflow(q)
        self.assertEqual(wf, "agent_trace_observability")

    def test_invoice_processing_signature_match(self):
        q = invoice_processing_questions()
        wf = match_typed_decisions_workflow(q)
        self.assertEqual(wf, "invoice_processing")

    def test_security_incidents_signature_match(self):
        q = security_incident_questions()
        wf = match_typed_decisions_workflow(q)
        self.assertEqual(wf, "security_incidents")

if __name__ == "__main__":
    unittest.main()
