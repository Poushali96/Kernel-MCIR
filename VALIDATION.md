# Validation performed before packaging

The review artifact was checked using:

```bash
PYTHONPATH=src pytest -q
python scripts/validate_artifact.py
```

The packaged version passes the unit tests for:

- zero-strength aggregation convention;
- boundedness of informative scores;
- signed-total identity;
- constructive context dependence;
- exact duplicate symmetry under exhaustive uniform permutation averaging.

The controlled experiment scripts were also executed end-to-end before packaging. Runtime values from these local checks are not committed because timing is machine dependent.
