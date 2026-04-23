"""LGD Python wrapper — loads R beta-regression predictions at scoring time.

Intentionally thin: LGD is implemented in R (R/lgd_r/) because betareg is the
actuarially correct tool for [0,1]-bounded targets. This module is the bridge
that lets the Python-side ECL engine consume those predictions.
"""
