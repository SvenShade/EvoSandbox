--- /mnt/data/simple_spread_sharedbasis_swarmscout_clearance.py	2026-04-15 04:16:23.863370440 +0000
+++ /mnt/data/simple_spread_sharedbasis_swarmscout_clearance_occspread.py	2026-04-15 04:17:24.895086114 +0000
@@ -1142,6 +1142,17 @@
         mid = aerials.shape[-1] // 2
         hgt = aerials[:, mid, mid]
         roi = hgt > 0.0
+        local_occupation = jnp.clip(jnp.where(roi, PRE(1.0) - jnp.abs(hgt), PRE(0.0)), PRE(0.0), PRE(1.0))
+
+        expl_side = state.expl_map.shape[-1]
+        roi_expl = jax.image.resize(state.hmap[1].astype(PRE), (expl_side, expl_side), "linear")
+        roi_expl = (roi_expl > PRE(0.5)).astype(PRE)
+        roi_mass = jnp.sum(roi_expl) + self.eps
+        team_roi_coverage = jnp.clip(
+            jnp.sum(state.expl_map * roi_expl[None, :, :], axis=(1, 2)) / roi_mass,
+            PRE(0.0),
+            PRE(1.0),
+        )
         
         # Per-agent reward function.
         # Rewards are largely local/individual and positive.
@@ -1175,10 +1186,9 @@
             scouting_term = scouting_terms[i].reshape(())
 
             # OCCUPATION --------------------------------------------------- #
-            # Is this agent directly overhead a ROI?
-            # TODO: Add team_roi_coverage
-            # occup = overhead_roi * dist_from_ground * team_roi_coverage
-            occupation_term = roi[i] * (1 - jnp.abs(hgt[i])) #* roi_coverage[i]
+            # Blend a local overhead-ROI term with a global team ROI coverage
+            # term derived from the exploration map. Both are in [0, 1].
+            occupation_term = PRE(0.5) * (local_occupation[i] + team_roi_coverage[team])
 
             # CONSERVATION ------------------------------------------------- #
 
