# Suggested future work history

Commit the actual initial implementation on the actual date. Do not split an
already completed implementation into misleading historical milestones or backdate it.
For subsequent work, commit when each change is made and reviewed:

1. Document an observed failure with a regression test and an issue.
2. Implement one improvement and compare its evaluation against this baseline.
3. Add a reviewed dataset slice and explain its provenance and split rules.
4. Add deployment configuration; publish only after the service is tested.
5. Publish a release with the exact commit, configuration and measured results.

Suggested messages: `test: reproduce out-of-domain failure`,
`fix: calibrate abstention on validation data`, `docs: publish held-out evaluation`.
These are future examples, not claims about work already performed.
