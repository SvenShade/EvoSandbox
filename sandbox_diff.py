--- simple_spread_sharedbasis_swarmscout_clearance_occspread.py	2026-04-16 05:49:32.002979267 +0000
+++ simple_spread_sharedbasis_swarmscout_clearance_occspread_pgonsoft.py	2026-04-16 05:54:16.746541997 +0000
@@ -526,30 +526,26 @@
         if self.num_pgon == 0:
             return state.pg_pos, state.pg_vel
 
-        # Flee away from up to the 3 nearest visible agents for each pigeon.
-        # pg_diff_pos is [A, P, 3], with invisible / invalid interactions zero-masked.
-        max_threats = 3
-        pg_diff = jnp.swapaxes(pg_diff_pos, 0, 1)      # [P, A, 3]
-        # Invalid / invisible interactions are zero-masked upstream; convert those
-        # zeros to +inf before nearest-neighbour selection so they are not chosen as
-        # spurious 'closest' threats.
-        pg_sqr = jnp.swapaxes(
-            jnp.where(pg_sqr_dist > 0.0, pg_sqr_dist, jnp.inf),
-            0,
-            1,
-        )       # [P, A]
-        pad_agents = max(0, max_threats - self.num_agnt)
-        pg_sqr_pad = jnp.pad(pg_sqr, ((0, 0), (0, pad_agents)), constant_values=jnp.inf)
-        pg_diff_pad = jnp.pad(pg_diff, ((0, 0), (0, pad_agents), (0, 0)))
-        _, near_idxs = jax.lax.top_k(-pg_sqr_pad, max_threats)
-        near_sqr = jnp.take_along_axis(pg_sqr_pad, near_idxs, axis=-1)
-        near_diff = jnp.take_along_axis(pg_diff_pad, near_idxs[..., None], axis=1)
-        near_valid = jnp.isfinite(near_sqr) & (near_sqr > 0.0)
+        # Flee from all valid visible agents using a masked inverse-distance
+        # weighted sum. This avoids the compiler-hostile top_k / inf / gather
+        # path while still strongly biasing motion away from nearby threats.
+        pg_diff = jnp.swapaxes(pg_diff_pos, 0, 1)   # [P, A, 3]
+        pg_sqr = jnp.swapaxes(pg_sqr_dist, 0, 1)    # [P, A]
+        valid = pg_sqr > 0.0
 
-        d = jnp.sqrt(jnp.maximum(near_sqr, 0.0)) + self.eps
-        away = near_diff / d[..., None]
-        away_w = 1.0 / d[..., None]
-        flee = jnp.sum(jnp.where(near_valid[..., None], away * away_w, 0.0), axis=1)
+        d = jnp.sqrt(jnp.maximum(pg_sqr, 0.0)) + self.eps
+        away = pg_diff / d[..., None]
+        weights = jnp.where(valid, 1.0 / d, 0.0)
+        flee = jnp.sum(away * weights[..., None], axis=1)
+
+        # Bound flee magnitude so many visible agents do not create excessive
+        # acceleration spikes.
+        flee_norm = jnp.linalg.norm(flee, axis=-1, keepdims=True)
+        flee = jnp.where(
+            flee_norm > 1.0,
+            flee / (flee_norm + self.eps),
+            flee,
+        )
 
         # Push up when too close to terrain. Keep a little lift to counter gravity.
         alt = state.pg_pos[:, -1] / self.env_size - pg_land_heights
