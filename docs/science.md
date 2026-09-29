# Science

The science behind the analyses that Ratatosk performs is described below.

---

# What a dQ/dV curve is

A galvanostatic cycle gives you charge **Q** and voltage **V**. Differentiating
one against the other, **dQ/dV**, converts a voltage plateau into a peak — and
plateaus are where the interesting electrochemistry is, because a plateau means
the potential is being *held* by something.

At constant current there is an identity worth keeping in mind, because most of
the artefacts in this pipeline come from it:

```
    dQ/dV  =  I / (dV/dt)          so at constant current,
    sign(dQ/dV) = sign(dV/dt)
```

The derivative is therefore only as good as the voltage record. Where the cell
sits on a flat plateau, dV between consecutive records is small; where it is
smaller than the instrument's voltage resolution, dQ/dV is not a measurement of
anything. This is the single largest limitation of the technique and §A4
returns to it.

**Three properties of a peak, and they are not equally trustworthy:**

| Property         | What it means                                                                                              | How much to trust it                                                                                           |
| ---------------- | ---------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **Position** (V) | The potential of a redox process. Shifts as polarisation grows.                                            | **Robust.** Does not involve the integral at all.                                                              |
| **Area**         | Charge associated with that process — *if* the curve accounts for the charge.                              | **Conditional.** See §A4 (does the curve account for the cell?) and §A5 (do the peaks account for the curve?). |
| **Width** (FWHM) | Homogeneity and kinetics of the process; a broader peak is a less well-defined transition or a slower one. | Moderate. Correlated with amplitude in any fit.                                                                |

### How the curve is computed, and why that is not a detail

There is **no widely agreed method** for computing an IC curve — one review
found nineteen studies with nineteen different workflows. Two families exist:

**Numerical differentiation** (the conventional method; Ratatosk offers it,
but it is *not* the default). Finite differences on the voltage array, with
Savitzky–Golay smoothing before differentiation. Because ΔV sits in the
denominator, flat plateaus give very large IC values, so smoothing is
essential — and its parameters are usually chosen by trial and error until the
curve *looks* right. That makes curves hard to compare between studies, and
the smoothing itself introduces drifts, oscillations and boundary overshoots.
Ratatosk at least chooses the parameters from a measured profile class rather
than by eye, and records every one of them; that does not escape the
underlying fragility, which is why it is not the path taken.

**Voltage histograms** (derivative-free — **what Ratatosk does**,
`DQDV_COMPUTATION = "histogram"` in Cell 2). Bin the records by voltage and
**sum the charge that passed inside each bin**, then divide by the bin width.
That is the IC curve, and nothing is ever differentiated. One parameter, the
bin size, which means exactly what it says: the voltage resolution of the
result.

A note on that, because the textbook statement of the method is weaker than
the implementation. The usual formulation *counts* the records that land in
each bin, which is proportional to the charge only if capacity is sampled at
uniform intervals — a condition real cycler files do not always meet.
Ratatosk passes the charge itself as the histogram weight
(`np.histogram(..., weights=dq)`), so the bin value is the charge that passed
there whether or not the sampling was uniform, and the assumption is not
needed. §A4 measures what this buys.

**The objective test between them is voltage reconstruction.** A derivative is
reversible, so the cumulative integral of the IC curve, plotted against V,
should give back the original voltage curve. Two metrics quantify how well:
the absolute percentage error in maximum cumulative capacity (APE) and the
voltage RMSE over the overlapping interval. Many parameter choices give a
visually satisfying curve; few reconstruct the voltage. Ratatosk's
`integral_fidelity` is the scalar special case of the capacity half of this
test — see §A4, where it matters a great deal.

**Broad features, or none at all, are not a failure.** A material undergoing a
continuous solid solution reaction has no distinct phase transition to hold the
potential, so it gives broad humps. Overlapping reactions merge. High rates
flatten and broaden everything. Coarse voltage sampling hides fine structure. A
featureless dQ/dV curve is a result about the material, not a problem with the
measurement.

> **Tuning.** The bin width and smoothing half-width are the two knobs on this
> section, both set from the measured profile class. If your material is
> broader or sharper than the classifier decided, correct the class first — it
> moves several parameters coherently. §A11 has the table, and the entries for
> *"curve is sparse"*, *"fine structure smeared"* and *"noisy and spiky"*.
