theory Mamba_Invariant
  imports Complex_Main
begin

section \<open>A machine-checked forward invariant for the selective SSM state\<close>

text \<open>
  This theory machine-checks the invariant argument of @{file \<open>docs/02-invariant.md\<close>} and
  the LayerNorm bounds of @{file \<open>docs/03-certificate.md\<close>}: the hidden state of a Mamba
  block with exact zero-order-hold discretisation never leaves the box \<open>[-M, M]\<close> with
  \<open>M = sup |B u| / |A|\<close>, at any sequence length, with no lower bound on the timescale.

  The five statements listed in the project brief are covered as follows.

    \<^item> The weighted-average lemma           \<rightarrow> @{text convex_combination_bound}
    \<^item> \<open>Abar = exp(dt * A) \<in> (0, 1)\<close>        \<rightarrow> @{text Abar_in_unit_interval}
    \<^item> The zero-order-hold identity          \<rightarrow> @{text zoh_identity}
    \<^item> The induction over \<open>t\<close>                \<rightarrow> @{text convex_recurrence_invariant},
                                              headline: @{text mamba_cell_state_bounded}
    \<^item> LayerNorm and fused bounds, general d \<rightarrow> @{text layernorm_bound},
                                              @{text fused_layernorm_linear_bound},
                                              @{text fused_layernorm_linear_attained}

  Everything is stated per coordinate: \<open>A\<close> is diagonal, so one cell is one scalar
  recurrence, and the block is 2048 independent copies of it. All statements are over the
  reals; floating point is out of scope here and remains the separate work item the brief
  lists as a bonus.
\<close>

section \<open>1. The weighted-average lemma\<close>

text \<open>
  A convex combination of two numbers in \<open>[-M, M]\<close> stays in \<open>[-M, M]\<close>. This is the whole
  mechanism: the coffee can never get hotter than the room. Stated on the closed interval
  \<open>a \<in> [0, 1]\<close> deliberately \<comment> \<open>the open interval is what the exact model produces, but the
  closed one is what the induction needs, and it additionally covers the two degenerate
  limits: \<open>a = 1\<close> (the update keeps the state; this is also where a float32 softplus
  underflow \<open>delta = 0\<close> lands) and \<open>a = 0\<close> (the state is flushed to the target).\<close>
\<close>

lemma convex_combination_bound:
  fixes a h c M :: real
  assumes "0 \<le> a" and "a \<le> 1" and "\<bar>h\<bar> \<le> M" and "\<bar>c\<bar> \<le> M"
  shows "\<bar>a * h + (1 - a) * c\<bar> \<le> M"
proof -
  have "\<bar>a * h + (1 - a) * c\<bar> \<le> a * \<bar>h\<bar> + (1 - a) * \<bar>c\<bar>"
    using assms by (simp add: abs_mult abs_triangle_ineq[THEN order_trans] add_mono)
  also have "\<dots> \<le> a * M + (1 - a) * M"
    using assms by (intro add_mono mult_left_mono) auto
  also have "\<dots> = M"
    by (simp add: algebra_simps)
  finally show ?thesis .
qed

section \<open>2. The discretised transition lies in the unit interval\<close>

text \<open>
  \<open>Abar = exp(dt * A)\<close> is a genuine convex weight for every timescale the selectivity
  mechanism can emit: strictly inside \<open>(0, 1)\<close> for \<open>dt > 0\<close>, and still inside \<open>(0, 1]\<close>
  when \<open>dt\<close> is only nonneg \<comment> \<open>the closed variant is the robustness margin for \<open>dt = 0\<close>.\<close>
\<close>

lemma Abar_in_unit_interval:
  fixes dt A :: real
  assumes "0 < dt" and "A < 0"
  shows "0 < exp (dt * A)" and "exp (dt * A) < 1"
proof -
  show "0 < exp (dt * A)" by simp
  have "dt * A < 0" using assms by (simp add: mult_pos_neg)
  then show "exp (dt * A) < 1" using exp_less_one_iff by simp
qed

lemma Abar_in_unit_interval_closed:
  fixes dt A :: real
  assumes "0 \<le> dt" and "A < 0"
  shows "0 < exp (dt * A)" and "exp (dt * A) \<le> 1"
proof -
  show "0 < exp (dt * A)" by simp
  have "dt * A \<le> 0" using assms by (simp add: mult_nonneg_nonpos)
  then show "exp (dt * A) \<le> 1" by simp
qed

section \<open>3. The zero-order-hold identity and the convex form\<close>

text \<open>
  Proposition 1 of @{file \<open>docs/02-invariant.md\<close>}: for \<open>A < 0\<close> the exact discretisation
  puts \<open>1 - Abar\<close> in the input gain, so the update is a weighted average. The Euler
  surrogate \<open>Bbar \<approx> dt * B\<close> does not satisfy this identity, which is why the invariant
  belongs to the exact discretisation only.
\<close>

lemma zoh_identity:
  fixes dt A B :: real
  assumes "A < 0"
  shows "(exp (dt * A) - 1) / A * B = (1 - exp (dt * A)) * B / \<bar>A\<bar>"
proof -
  have "\<bar>A\<bar> = - A" using assms by simp
  then show ?thesis using assms by (simp add: field_simps)
qed

lemma update_is_convex_combination:
  fixes dt A B u h :: real
  assumes "A < 0"
  defines "Abar \<equiv> exp (dt * A)" and "c \<equiv> B * u / \<bar>A\<bar>"
  shows "Abar * h + (exp (dt * A) - 1) / A * B * u = Abar * h + (1 - Abar) * c"
  using assms by (simp add: field_simps)

text \<open>
  The name "exact" is itself a theorem: the discrete step is the flow of the leaky
  integrator \<open>h' = A h + b\<close> (with \<open>b = B u\<close> held constant over the tick). The candidate
  \<open>s \<mapsto> exp(s A) h0 + (exp(s A) - 1)/A \<cdot> b\<close> starts at \<open>h0\<close> and satisfies the ODE, so the
  recurrence below is Newton's law of cooling sampled at the ticks, not an approximation
  of it. (Uniqueness of the flow is classical Picard--LindelÃ\<paragraph>f and is not needed: the
  discrete recurrence is the verified object, and this lemma only certifies its pedigree.)
\<close>

lemma zoh_is_exact_flow:
  fixes A b h0 s :: real
  assumes "A \<noteq> 0"
  shows "((\<lambda>s. exp (s * A) * h0 + (exp (s * A) - 1) / A * b) has_field_derivative
          A * (exp (s * A) * h0 + (exp (s * A) - 1) / A * b) + b) (at s)"
    and "exp (0 * A) * h0 + (exp (0 * A) - 1) / A * b = h0"
proof -
  have "((\<lambda>s. exp (s * A) * h0 + (exp (s * A) - 1) / A * b) has_field_derivative
         exp (s * A) * A * h0 + exp (s * A) * A / A * b) (at s)"
    by (auto intro!: derivative_eq_intros)
  moreover have "exp (s * A) * A * h0 + exp (s * A) * A / A * b
      = A * (exp (s * A) * h0 + (exp (s * A) - 1) / A * b) + b"
    using assms by (simp add: field_simps)
  ultimately show "((\<lambda>s. exp (s * A) * h0 + (exp (s * A) - 1) / A * b) has_field_derivative
          A * (exp (s * A) * h0 + (exp (s * A) - 1) / A * b) + b) (at s)"
    by simp
  show "exp (0 * A) * h0 + (exp (0 * A) - 1) / A * b = h0"
    by simp
qed

section \<open>4. The induction over time\<close>

text \<open>
  One step applied \<open>t\<close> times. The index convention shifts the paper's \<open>h\<^sub>-\<^sub>1 = 0\<close> to
  \<open>h 0 = 0\<close>: \<open>h t\<close> is the state after \<open>t\<close> ticks, and the coefficients consumed at tick
  \<open>t+1\<close> carry index \<open>t\<close>. The general form allows any initial state and yields
  \<open>max \<bar>h 0\<bar> M\<close>, so a deployment that carries state across windows stays certified: once
  inside the box, forever inside the box.
\<close>

lemma convex_recurrence_bounded:
  fixes a c h :: "nat \<Rightarrow> real" and M :: real
  assumes step: "\<And>t. h (Suc t) = a t * h t + (1 - a t) * c t"
    and weight_lo: "\<And>t. 0 \<le> a t" and weight_hi: "\<And>t. a t \<le> 1"
    and target: "\<And>t. \<bar>c t\<bar> \<le> M"
  shows "\<bar>h t\<bar> \<le> max \<bar>h 0\<bar> M"
proof (induction t)
  case 0
  then show ?case by simp
next
  case (Suc t)
  have "\<bar>c t\<bar> \<le> max \<bar>h 0\<bar> M" using target[of t] by simp
  then have "\<bar>a t * h t + (1 - a t) * c t\<bar> \<le> max \<bar>h 0\<bar> M"
    using Suc.IH weight_lo[of t] weight_hi[of t] by (intro convex_combination_bound)
  then show ?case by (simp add: step)
qed

corollary convex_recurrence_invariant:
  fixes a c h :: "nat \<Rightarrow> real" and M :: real
  assumes "h 0 = 0"
    and "\<And>t. h (Suc t) = a t * h t + (1 - a t) * c t"
    and "\<And>t. 0 \<le> a t" and "\<And>t. a t \<le> 1"
    and "\<And>t. \<bar>c t\<bar> \<le> M"
  shows "\<bar>h t\<bar> \<le> M"
proof -
  have "0 \<le> M" using assms(5)[of 0] by simp
  moreover have "\<bar>h t\<bar> \<le> max \<bar>h 0\<bar> M"
    using assms(2-5) by (rule convex_recurrence_bounded)
  ultimately show ?thesis using assms(1) by simp
qed

text \<open>
  The headline theorem, assembled from the pieces. The hypotheses are exactly what the
  architecture supplies: \<open>A < 0\<close> from the parametrisation (Section 6), \<open>dt > 0\<close> from
  either timescale arm (Section 6), and \<open>\<bar>B u\<bar> \<le> K\<close> from the certified boxes whose
  LayerNorm root is Section 5. The recurrence hypothesis is the exact zero-order hold as
  written, before any rearrangement. The conclusion holds for every \<open>t\<close> \<comment> \<open>the bound
  \<open>K / \<bar>A\<bar>\<close> mentions no sequence length, no timescale floor, and no unrolling.\<close>
\<close>

theorem mamba_cell_state_bounded:
  fixes A K :: real and dt B u h :: "nat \<Rightarrow> real"
  assumes A_neg: "A < 0"
    and dt_pos: "\<And>t. 0 < dt t"
    and drive: "\<And>t. \<bar>B t * u t\<bar> \<le> K"
    and init: "h 0 = 0"
    and step: "\<And>t. h (Suc t) =
                 exp (dt t * A) * h t + (exp (dt t * A) - 1) / A * B t * u t"
  shows "\<bar>h t\<bar> \<le> K / \<bar>A\<bar>"
proof -
  define a where "a \<equiv> \<lambda>t. exp (dt t * A)"
  define c where "c \<equiv> \<lambda>t. B t * u t / \<bar>A\<bar>"
  have conv_step: "h (Suc t) = a t * h t + (1 - a t) * c t" for t
    using step[of t] update_is_convex_combination[OF A_neg, of "dt t" "h t" "B t" "u t"]
    by (simp add: a_def c_def)
  have a_lo: "0 \<le> a t" for t
    unfolding a_def by simp
  have a_hi: "a t \<le> 1" for t
    unfolding a_def using Abar_in_unit_interval[OF dt_pos A_neg] by (simp add: less_imp_le)
  have c_bound: "\<bar>c t\<bar> \<le> K / \<bar>A\<bar>" for t
  proof -
    have "\<bar>c t\<bar> = \<bar>B t * u t\<bar> / \<bar>A\<bar>"
      unfolding c_def by simp
    also have "\<dots> \<le> K / \<bar>A\<bar>"
      using drive[of t] by (intro divide_right_mono) auto
    finally show ?thesis .
  qed
  show ?thesis
    by (rule convex_recurrence_invariant[of h a c "K / \<bar>A\<bar>"])
       (use init conv_step a_lo a_hi c_bound in auto)
qed

section \<open>5. The LayerNorm bounds, at the model's width and every other\<close>

text \<open>
  Z3 proves these at \<open>d \<le> 5\<close> and times out beyond; the model has \<open>d = 64\<close>. Here they are
  for every finite width, over an arbitrary finite index set. Complex_Main carries no
  Cauchy--Schwarz for finite sums, so it is proved first, by the discriminant argument.
\<close>

lemma cauchy_schwarz_sum:
  fixes f g :: "'a \<Rightarrow> real" and I :: "'a set"
  shows "(\<Sum>i\<in>I. f i * g i)\<^sup>2 \<le> (\<Sum>i\<in>I. (f i)\<^sup>2) * (\<Sum>i\<in>I. (g i)\<^sup>2)"
proof (cases "finite I \<and> (\<Sum>i\<in>I. (g i)\<^sup>2) \<noteq> 0")
  case False
  then show ?thesis
  proof (cases "finite I")
    case True
    with False have g0: "(\<Sum>i\<in>I. (g i)\<^sup>2) = 0" by blast
    then have "\<And>i. i \<in> I \<Longrightarrow> g i = 0"
      using True by (simp add: sum_nonneg_eq_0_iff)
    then have "(\<Sum>i\<in>I. f i * g i) = 0" by simp
    then show ?thesis using g0 by simp
  next
    case False
    then show ?thesis by simp
  qed
next
  case True
  then have fin: "finite I" and gnz: "(\<Sum>i\<in>I. (g i)\<^sup>2) \<noteq> 0" by auto
  define Sg where "Sg = (\<Sum>i\<in>I. (g i)\<^sup>2)"
  define t where "t = (\<Sum>i\<in>I. f i * g i) / Sg"
  have Sg_pos: "0 < Sg"
    unfolding Sg_def using gnz sum_nonneg[of I "\<lambda>i. (g i)\<^sup>2"]
    by (auto simp: less_le)
  have "0 \<le> (\<Sum>i\<in>I. (f i - t * g i)\<^sup>2)"
    by (simp add: sum_nonneg)
  also have "(\<Sum>i\<in>I. (f i - t * g i)\<^sup>2)
      = (\<Sum>i\<in>I. (f i)\<^sup>2) - 2 * t * (\<Sum>i\<in>I. f i * g i) + t\<^sup>2 * Sg"
    unfolding Sg_def
    by (simp add: power2_diff sum.distrib sum_subtractf sum_distrib_left
                  algebra_simps)
  finally have "0 \<le> (\<Sum>i\<in>I. (f i)\<^sup>2) - (\<Sum>i\<in>I. f i * g i)\<^sup>2 / Sg"
    using Sg_pos by (simp add: t_def power2_eq_square field_simps)
  then have "(\<Sum>i\<in>I. f i * g i)\<^sup>2 / Sg \<le> (\<Sum>i\<in>I. (f i)\<^sup>2)" by simp
  then show ?thesis
    using Sg_pos unfolding Sg_def by (simp add: divide_le_eq)
qed

text \<open>
  The coordinate bound of @{file \<open>docs/03-certificate.md\<close>}: on the LayerNorm constraint
  set (zero sum, energy at most \<open>d\<close>) no coordinate exceeds \<open>sqrt (d - 1)\<close>. To push one
  coordinate high, the other \<open>d - 1\<close> must cancel it, and the cancellation spends energy.
  The statement degenerates gracefully at \<open>d = 1\<close>: the constraints force \<open>z = 0\<close>.
\<close>

lemma layernorm_coordinate_bound_sq:
  fixes z :: "'a \<Rightarrow> real" and I :: "'a set" and i :: 'a
  assumes fin: "finite I" and mem: "i \<in> I"
    and zsum: "(\<Sum>j\<in>I. z j) = 0"
    and znorm: "(\<Sum>j\<in>I. (z j)\<^sup>2) \<le> real (card I)"
  shows "(z i)\<^sup>2 \<le> real (card I) - 1"
proof -
  define J where "J = I - {i}"
  define D where "D = real (card I)"
  have card_pos: "0 < card I"
    using fin mem by (auto simp: card_gt_0_iff)
  have D_ge1: "1 \<le> D"
    unfolding D_def using card_pos by simp
  have "card J = card I - 1"
    unfolding J_def using mem by (simp add: card_Diff_singleton)
  then have cardJ: "real (card J) = D - 1"
    unfolding D_def using card_pos by (simp add: Suc_leI)
  have sumJ: "(\<Sum>j\<in>J. z j) = - z i"
    unfolding J_def using fin mem zsum by (simp add: sum_diff1)
  have sqJ: "(\<Sum>j\<in>J. (z j)\<^sup>2) = (\<Sum>j\<in>I. (z j)\<^sup>2) - (z i)\<^sup>2"
    unfolding J_def using fin mem by (simp add: sum_diff1)
  have "(z i)\<^sup>2 = (\<Sum>j\<in>J. z j * 1)\<^sup>2"
    using sumJ by simp
  also have "\<dots> \<le> (\<Sum>j\<in>J. (z j)\<^sup>2) * (\<Sum>j\<in>J. (1::real)\<^sup>2)"
    by (rule cauchy_schwarz_sum)
  also have "(\<Sum>j\<in>J. (1::real)\<^sup>2) = real (card J)"
    by simp
  finally have cs: "(z i)\<^sup>2 \<le> (\<Sum>j\<in>J. (z j)\<^sup>2) * (D - 1)"
    by (simp add: cardJ)
  have EJ: "(\<Sum>j\<in>J. (z j)\<^sup>2) \<le> D - (z i)\<^sup>2"
    using sqJ znorm unfolding D_def by simp
  have key: "(z i)\<^sup>2 \<le> (D - (z i)\<^sup>2) * (D - 1)"
    using cs EJ D_ge1 mult_right_mono[OF EJ, of "D - 1"] by simp
  have expand: "(D - (z i)\<^sup>2) * (D - 1) = D * D - D - (z i)\<^sup>2 * D + (z i)\<^sup>2"
    by (simp add: algebra_simps)
  from key have "0 \<le> D * D - D - (z i)\<^sup>2 * D"
    unfolding expand by linarith
  also have "D * D - D - (z i)\<^sup>2 * D = D * (D - 1 - (z i)\<^sup>2)"
    by (simp add: algebra_simps)
  finally have "0 \<le> D - 1 - (z i)\<^sup>2"
    using D_ge1 by (auto simp: zero_le_mult_iff)
  then show ?thesis
    unfolding D_def by linarith
qed

corollary layernorm_coordinate_bound:
  fixes z :: "'a \<Rightarrow> real" and I :: "'a set" and i :: 'a
  assumes "finite I" and "i \<in> I"
    and "(\<Sum>j\<in>I. z j) = 0"
    and "(\<Sum>j\<in>I. (z j)\<^sup>2) \<le> real (card I)"
  shows "\<bar>z i\<bar> \<le> sqrt (real (card I) - 1)"
proof -
  have "(z i)\<^sup>2 \<le> real (card I) - 1"
    by (rule layernorm_coordinate_bound_sq[OF assms])
  then have "sqrt ((z i)\<^sup>2) \<le> sqrt (real (card I) - 1)"
    by (rule real_sqrt_le_mono)
  then show ?thesis
    by simp
qed

text \<open>
  The constraint set is not an assumption on the data: the normalisation itself lands in
  it, for every input \<open>x \<in> \<real>\<^sup>d\<close> and every \<open>eps > 0\<close>. This is the full Proposition of
  @{file \<open>docs/03-certificate.md\<close>}, \<open>\<bar>LN(x)\<^sub>i - beta\<^sub>i\<bar> \<le> \<bar>gamma\<^sub>i\<bar> sqrt (d - 1)\<close>, with the
  biased variance PyTorch uses. The architecture manufactures the invariant's premise.
\<close>

theorem layernorm_bound:
  fixes x gamma beta :: "'a \<Rightarrow> real" and I :: "'a set" and i :: 'a and eps :: real
  assumes fin: "finite I" and mem: "i \<in> I" and eps_pos: "0 < eps"
  defines "d \<equiv> real (card I)"
    and "mu \<equiv> (\<Sum>j\<in>I. x j) / real (card I)"
  defines "var \<equiv> (\<Sum>j\<in>I. (x j - mu)\<^sup>2) / real (card I)"
  defines "z \<equiv> \<lambda>j. (x j - mu) / sqrt (var + eps)"
  defines "LN \<equiv> \<lambda>j. gamma j * z j + beta j"
  shows "\<bar>LN i - beta i\<bar> \<le> \<bar>gamma i\<bar> * sqrt (d - 1)"
proof -
  have card_pos: "0 < card I"
    using fin mem by (auto simp: card_gt_0_iff)
  have d_pos: "0 < d"
    unfolding d_def using card_pos by simp
  have var_nonneg: "0 \<le> var"
    unfolding var_def by (simp add: sum_nonneg)
  have veps_pos: "0 < var + eps"
    using var_nonneg eps_pos by simp
  have centred_sum: "(\<Sum>j\<in>I. x j - mu) = 0"
    using d_pos unfolding mu_def d_def by (simp add: sum_subtractf)
  have zsum: "(\<Sum>j\<in>I. z j) = 0"
    unfolding z_def by (simp flip: sum_divide_distrib add: centred_sum)
  have centred_sq: "(\<Sum>j\<in>I. (x j - mu)\<^sup>2) = d * var"
    using d_pos unfolding var_def d_def by simp
  have z_sq_pointwise: "(z j)\<^sup>2 = (x j - mu)\<^sup>2 / (var + eps)" for j
    unfolding z_def using veps_pos by (simp add: power_divide)
  have z_sq: "(\<Sum>j\<in>I. (z j)\<^sup>2) = d * var / (var + eps)"
    by (simp add: z_sq_pointwise flip: sum_divide_distrib) (simp add: centred_sq)
  have "d * var / (var + eps) \<le> d"
    using d_pos var_nonneg eps_pos
    by (simp add: divide_le_eq mult_left_mono)
  with z_sq have znorm: "(\<Sum>j\<in>I. (z j)\<^sup>2) \<le> real (card I)"
    unfolding d_def by simp
  have "\<bar>z i\<bar> \<le> sqrt (real (card I) - 1)"
    by (rule layernorm_coordinate_bound[OF fin mem zsum znorm])
  then have "\<bar>gamma i\<bar> * \<bar>z i\<bar> \<le> \<bar>gamma i\<bar> * sqrt (real (card I) - 1)"
    by (intro mult_left_mono) auto
  then show ?thesis
    unfolding LN_def d_def by (simp add: abs_mult)
qed

text \<open>
  The fused LayerNorm-and-Linear bound: over the constraint set, \<open>\<langle>v, z\<rangle>\<close> is bounded by
  \<open>\<parallel>v - mean v\<parallel>\<^sub>2 sqrt d\<close> \<comment> \<open>the zero-sum constraint subtracts the mean of the weight row
  for free, an \<open>l\<^sub>2\<close> quantity where boxing the LayerNorm first pays an \<open>l\<^sub>1\<close> one. This is
  the 49x on the certified radius.\<close> Soundness first, then attainment: the maximiser lies
  in the set and meets the bound with equality, so the fused bound cannot be improved.
\<close>

lemma fused_layernorm_linear_bound:
  fixes v z :: "'a \<Rightarrow> real" and I :: "'a set"
  assumes fin: "finite I"
    and zsum: "(\<Sum>j\<in>I. z j) = 0"
    and znorm: "(\<Sum>j\<in>I. (z j)\<^sup>2) \<le> real (card I)"
  defines "vbar \<equiv> (\<Sum>j\<in>I. v j) / real (card I)"
  shows "\<bar>\<Sum>j\<in>I. v j * z j\<bar> \<le> sqrt (\<Sum>j\<in>I. (v j - vbar)\<^sup>2) * sqrt (real (card I))"
proof -
  have "(\<Sum>j\<in>I. (v j - vbar) * z j) = (\<Sum>j\<in>I. v j * z j) - vbar * (\<Sum>j\<in>I. z j)"
    by (simp add: left_diff_distrib sum_subtractf flip: sum_distrib_left)
  then have shift: "(\<Sum>j\<in>I. v j * z j) = (\<Sum>j\<in>I. (v j - vbar) * z j)"
    by (simp add: zsum)
  have "(\<Sum>j\<in>I. (v j - vbar) * z j)\<^sup>2 \<le> (\<Sum>j\<in>I. (v j - vbar)\<^sup>2) * (\<Sum>j\<in>I. (z j)\<^sup>2)"
    by (rule cauchy_schwarz_sum)
  also have "\<dots> \<le> (\<Sum>j\<in>I. (v j - vbar)\<^sup>2) * real (card I)"
    using znorm by (intro mult_left_mono) (auto simp: sum_nonneg)
  finally have sq: "(\<Sum>j\<in>I. v j * z j)\<^sup>2 \<le> (\<Sum>j\<in>I. (v j - vbar)\<^sup>2) * real (card I)"
    using shift by simp
  have "\<bar>\<Sum>j\<in>I. v j * z j\<bar> = sqrt ((\<Sum>j\<in>I. v j * z j)\<^sup>2)"
    by simp
  also have "\<dots> \<le> sqrt ((\<Sum>j\<in>I. (v j - vbar)\<^sup>2) * real (card I))"
    using sq by (rule real_sqrt_le_mono)
  also have "\<dots> = sqrt (\<Sum>j\<in>I. (v j - vbar)\<^sup>2) * sqrt (real (card I))"
    by (simp add: real_sqrt_mult)
  finally show ?thesis .
qed

lemma fused_layernorm_linear_attained:
  fixes v :: "'a \<Rightarrow> real" and I :: "'a set"
  assumes fin: "finite I" and ne: "I \<noteq> {}"
  defines "vbar \<equiv> (\<Sum>j\<in>I. v j) / real (card I)"
  defines "N \<equiv> sqrt (\<Sum>j\<in>I. (v j - vbar)\<^sup>2)"
  assumes N_pos: "0 < N"
  defines "zstar \<equiv> \<lambda>j. sqrt (real (card I)) * (v j - vbar) / N"
  shows "(\<Sum>j\<in>I. zstar j) = 0"
    and "(\<Sum>j\<in>I. (zstar j)\<^sup>2) = real (card I)"
    and "(\<Sum>j\<in>I. v j * zstar j) = N * sqrt (real (card I))"
proof -
  have card_pos: "0 < card I"
    using fin ne by (auto simp: card_gt_0_iff)
  have Nsq: "N\<^sup>2 = (\<Sum>j\<in>I. (v j - vbar)\<^sup>2)"
    unfolding N_def by (simp add: sum_nonneg)
  have sum_pos: "0 < (\<Sum>j\<in>I. (v j - vbar)\<^sup>2)"
    using N_pos Nsq zero_less_power[of N 2] by simp
  have centred_sum: "(\<Sum>j\<in>I. v j - vbar) = 0"
    using card_pos unfolding vbar_def by (simp add: sum_subtractf)
  show "(\<Sum>j\<in>I. zstar j) = 0"
    unfolding zstar_def
    by (simp add: centred_sum flip: sum_divide_distrib sum_distrib_left)
  have "(\<Sum>j\<in>I. (zstar j)\<^sup>2)
      = real (card I) * (\<Sum>j\<in>I. (v j - vbar)\<^sup>2) / N\<^sup>2"
    unfolding zstar_def
    by (simp add: power_divide power_mult_distrib sum_nonneg
             flip: sum_divide_distrib sum_distrib_left)
  then show "(\<Sum>j\<in>I. (zstar j)\<^sup>2) = real (card I)"
    using N_pos Nsq sum_pos by simp
  have vdot: "(\<Sum>j\<in>I. v j * (v j - vbar)) = (\<Sum>j\<in>I. (v j - vbar)\<^sup>2)"
  proof -
    have "v j * (v j - vbar) = (v j - vbar)\<^sup>2 + vbar * (v j - vbar)" for j
      by (simp add: power2_eq_square algebra_simps)
    then have "(\<Sum>j\<in>I. v j * (v j - vbar))
        = (\<Sum>j\<in>I. (v j - vbar)\<^sup>2) + vbar * (\<Sum>j\<in>I. v j - vbar)"
      by (simp add: sum.distrib flip: sum_distrib_left)
    then show ?thesis by (simp add: centred_sum)
  qed
  have pointwise: "v j * zstar j = sqrt (real (card I)) * (v j * (v j - vbar)) / N" for j
    unfolding zstar_def by (simp add: mult_ac)
  have "(\<Sum>j\<in>I. v j * zstar j) = sqrt (real (card I)) * (\<Sum>j\<in>I. v j * (v j - vbar)) / N"
    by (simp add: pointwise flip: sum_divide_distrib sum_distrib_left)
  also have "\<dots> = sqrt (real (card I)) * N\<^sup>2 / N"
    by (simp add: vdot Nsq)
  also have "\<dots> = N * sqrt (real (card I))"
    using N_pos by (simp add: power2_eq_square)
  finally show "(\<Sum>j\<in>I. v j * zstar j) = N * sqrt (real (card I))" .
qed

section \<open>6. Premises manufactured by the parametrisation\<close>

text \<open>
  \<open>A < 0\<close> and \<open>dt > 0\<close> hold for every weight draw and every input, not because of
  training: \<open>A = -exp A_log\<close> is negative for any real \<open>A_log\<close>, the softplus arm
  \<open>ln (1 + exp z)\<close> is positive for any real \<open>z\<close>, and the bounded arm
  \<open>dt_min \<cdot> (dt_max/dt_min) powr s\<close> is positive whenever the two endpoints are. Together
  with @{text layernorm_bound} these discharge every hypothesis of
  @{text mamba_cell_state_bounded} by construction.
\<close>

lemma A_parametrisation_negative:
  fixes Alog :: real
  shows "- exp Alog < 0"
  by simp

lemma softplus_positive:
  fixes z :: real
  shows "0 < ln (1 + exp z)"
  using exp_gt_zero[of z] by (simp add: ln_gt_zero)

lemma bounded_dt_positive:
  fixes dt_min dt_max s :: real
  assumes "0 < dt_min" and "0 < dt_max"
  shows "0 < dt_min * (dt_max / dt_min) powr s"
  using assms by simp

section \<open>7. The abstraction gap is a genuine gap\<close>

text \<open>
  The closed-form overcharge of @{file \<open>docs/02-invariant.md\<close>},
  \<open>gap = (1 - exp(-lam d_hi)) / (1 - exp(-lam d_lo))\<close>, is at least \<open>1\<close> whenever the
  timescale box is nondegenerate: the geometric bound never beats the invariant, at any
  pole and any timescale range. (Its divergence as \<open>d_lo \<rightarrow> 0\<close> is visible in the formula;
  the invariant does not mention \<open>d_lo\<close> at all.)
\<close>

lemma abstraction_gap_ge_one:
  fixes lam d_lo d_hi :: real
  assumes lam_pos: "0 < lam" and lo_pos: "0 < d_lo" and le: "d_lo \<le> d_hi"
  shows "1 \<le> (1 - exp (- lam * d_hi)) / (1 - exp (- lam * d_lo))"
proof -
  have denom_pos: "0 < 1 - exp (- lam * d_lo)"
    using lam_pos lo_pos by simp
  have "exp (- lam * d_hi) \<le> exp (- lam * d_lo)"
    using lam_pos le by simp
  then have "1 - exp (- lam * d_lo) \<le> 1 - exp (- lam * d_hi)"
    by simp
  then show ?thesis
    using denom_pos by simp
qed

section \<open>Scope\<close>

text \<open>
  What this theory does not cover, so nobody reads more into it than is there.

    \<^item> Reals, not floats. The model runs in float32; in particular the reference
      implementation's \<open>phi\<close> uses a cubic Taylor branch below \<open>1e-4\<close>, whose real-valued
      deviation from the exact zero-order hold is of relative order \<open>4e-14\<close> on that
      branch (far below float32 resolution, but a statement about rounding belongs to
      the floating-point work item, not to this file).
    \<^item> One cell, all cells. Statements are per scalar cell; the block is their product,
      and no cell interacts with another. The elementwise reading is the model.
    \<^item> The concrete numbers are not re-derived here. \<open>M = 129.6739\<close> comes from interval
      propagation through the trained weights; this file proves the schema of that
      computation (the invariant, the LayerNorm root, the fused bound, tightness of the
      gap formula), while the arithmetic on the weights stays in
      @{file \<open>src/verification/invariant.py\<close>} and its Z3-checked interval lemmas.
    \<^item> Composition across blocks and the head remains interval arithmetic in code, as
      @{file \<open>docs/05-handoff.md\<close>} already states.
\<close>

end
