(* UNCHECKED DRAFT. This theory was written without access to Isabelle and has never been
   loaded or checked. Every proof method below is a guess at what will close the goal.
   It is not listed in ROOT. Build it against Mamba_Invariant before relying on any line. *)

theory Mamba_Sensitivity
  imports "../Mamba_Invariant"
begin

section \<open>Sensitivity of the selective SSM cell to perturbed timescales and targets\<close>

text \<open>
  Three statements. A scalar bound on how far the transition weight \<open>exp (-x)\<close> moves when the
  timescale is rescaled by \<open>exp s\<close>; the algebraic identity for the difference of two runs of
  the convex recurrence; and the cell bound obtained by feeding that difference back into
  @{text convex_recurrence_bounded}.
\<close>

section \<open>1. The scalar rescaling bound\<close>

lemma exp_neg_tangent_bound:
  fixes x :: real
  shows "x * exp (- x) \<le> 1 - exp (- x)"
proof -
  have "1 + x \<le> exp x"
    by (rule exp_ge_add_one_self)
  then have "(1 + x) * exp (- x) \<le> exp x * exp (- x)"
    by (rule mult_right_mono) simp
  also have "exp x * exp (- x) = 1"
    by (rule exp_minus_inverse)
  finally show ?thesis
    by (simp add: algebra_simps)
qed

lemma one_minus_exp_neg_div_antimono:
  fixes x y :: real
  assumes y_pos: "0 < y" and y_le: "y \<le> x"
  shows "(1 - exp (- x)) / x \<le> (1 - exp (- y)) / y"
proof -
  have deriv: "\<exists>D. ((\<lambda>z. (1 - exp (- z)) / z) has_field_derivative D) (at z) \<and> D \<le> 0"
    if "y \<le> z" "z \<le> x" for z
  proof -
    have z_pos: "0 < z"
      using y_pos that by linarith
    have "((\<lambda>z. (1 - exp (- z)) / z) has_field_derivative
            (z * exp (- z) - (1 - exp (- z))) / z\<^sup>2) (at z)"
      using z_pos by (auto intro!: derivative_eq_intros simp: field_simps power2_eq_square)
    moreover have "(z * exp (- z) - (1 - exp (- z))) / z\<^sup>2 \<le> 0"
      using exp_neg_tangent_bound[of z] z_pos by (intro divide_nonpos_pos) auto
    ultimately show ?thesis
      by blast
  qed
  show ?thesis
    by (rule DERIV_nonpos_imp_decreasing[of y x "\<lambda>z. (1 - exp (- z)) / z", OF y_le deriv])
qed

text \<open>
  The rescaling bound. For \<open>x > 0\<close> and any real \<open>s\<close>,
  \<open>\<bar>exp (-x) - exp (-(x exp s))\<bar> \<le> \<bar>exp s - 1\<bar> (1 - exp (-x))\<close>. For \<open>s \<ge> 0\<close> it follows from
  the tangent line of \<open>exp\<close> at \<open>-x\<close>; for \<open>s < 0\<close> it follows from the monotonicity of
  \<open>(1 - exp (-y)) / y\<close>.
\<close>

lemma exp_rescale_bound:
  fixes x s :: real
  assumes x_pos: "0 < x"
  shows "\<bar>exp (- x) - exp (- (x * exp s))\<bar> \<le> \<bar>exp s - 1\<bar> * (1 - exp (- x))"
proof (cases "0 \<le> s")
  case True
  let ?y = "x * exp s"
  have e_ge: "1 \<le> exp s"
    using True by simp
  have e_abs: "\<bar>exp s - 1\<bar> = exp s - 1"
    using abs_of_nonneg[of "exp s - 1"] e_ge by simp
  have y_ge: "x \<le> ?y"
    using mult_left_mono[OF e_ge, of x] x_pos by simp
  have nonneg: "0 \<le> exp (- x) - exp (- ?y)"
    using y_ge by simp
  have split: "exp (- ?y) = exp (- x) * exp (- (?y - x))"
    by (simp add: mult_exp_exp algebra_simps)
  have tangent: "1 - (?y - x) \<le> exp (- (?y - x))"
    using exp_ge_add_one_self[of "- (?y - x)"] by (simp add: algebra_simps)
  have lower: "exp (- x) * (1 - (?y - x)) \<le> exp (- ?y)"
    unfolding split by (rule mult_left_mono[OF tangent]) simp
  have lin: "exp (- x) * (1 - (?y - x)) = exp (- x) - exp (- x) * (?y - x)"
    by (simp add: algebra_simps)
  have "\<bar>exp (- x) - exp (- ?y)\<bar> = exp (- x) - exp (- ?y)"
    by (rule abs_of_nonneg[OF nonneg])
  also have "\<dots> \<le> exp (- x) * (?y - x)"
    using lower lin by linarith
  also have "\<dots> = (x * exp (- x)) * (exp s - 1)"
    by (simp add: algebra_simps)
  also have "\<dots> \<le> (1 - exp (- x)) * (exp s - 1)"
    using exp_neg_tangent_bound[of x] e_ge by (intro mult_right_mono) auto
  also have "\<dots> = \<bar>exp s - 1\<bar> * (1 - exp (- x))"
    by (simp add: e_abs mult.commute)
  finally show ?thesis .
next
  case False
  let ?y = "x * exp s"
  have e_lt: "exp s < 1"
    using False by simp
  have e_abs: "\<bar>exp s - 1\<bar> = 1 - exp s"
    using abs_of_neg[of "exp s - 1"] e_lt by simp
  have y_pos: "0 < ?y"
    using x_pos by simp
  have y_le: "?y \<le> x"
    using mult_left_mono[of "exp s" 1 x] e_lt x_pos by simp
  have mono: "(1 - exp (- x)) / x \<le> (1 - exp (- ?y)) / ?y"
    by (rule one_minus_exp_neg_div_antimono[OF y_pos y_le])
  have scaled: "?y * ((1 - exp (- x)) / x) \<le> ?y * ((1 - exp (- ?y)) / ?y)"
    using mono y_pos by (intro mult_left_mono) auto
  have eq1: "?y * ((1 - exp (- x)) / x) = exp s * (1 - exp (- x))"
    using x_pos by (simp add: field_simps)
  have eq2: "?y * ((1 - exp (- ?y)) / ?y) = 1 - exp (- ?y)"
    using y_pos by (simp add: field_simps)
  have lower: "exp s * (1 - exp (- x)) \<le> 1 - exp (- ?y)"
    using scaled eq1 eq2 by linarith
  have nonneg: "0 \<le> exp (- ?y) - exp (- x)"
    using y_le by simp
  have lin: "(1 - exp s) * (1 - exp (- x)) = 1 - exp (- x) - exp s + exp s * exp (- x)"
    by (simp add: algebra_simps)
  have lin2: "exp s * (1 - exp (- x)) = exp s - exp s * exp (- x)"
    by (simp add: algebra_simps)
  have "\<bar>exp (- x) - exp (- ?y)\<bar> = \<bar>exp (- ?y) - exp (- x)\<bar>"
    by (rule abs_minus_commute)
  also have "\<dots> = exp (- ?y) - exp (- x)"
    by (rule abs_of_nonneg[OF nonneg])
  also have "\<dots> \<le> (1 - exp s) * (1 - exp (- x))"
    using lower lin lin2 by linarith
  also have "\<dots> = \<bar>exp s - 1\<bar> * (1 - exp (- x))"
    by (simp add: e_abs)
  finally show ?thesis .
qed

section \<open>2. The error recursion for two runs\<close>

text \<open>
  For \<open>h' = a h + (1 - a) c\<close> and \<open>g' = b g + (1 - b) d\<close>, the difference satisfies
  \<open>h' - g' = a (h - g) + (1 - a) (c - d) + (a - b) (g - d)\<close>.
\<close>

lemma error_recursion:
  fixes a b h g c d :: real
  shows "(a * h + (1 - a) * c) - (b * g + (1 - b) * d)
           = a * (h - g) + (1 - a) * (c - d) + (a - b) * (g - d)"
  by (simp add: algebra_simps)

section \<open>3. The cell bound\<close>

text \<open>
  Two runs of the cell from arbitrary initial states. If the second run's weight is within
  \<open>k (1 - a t)\<close> of the first's, both its target and its state lie in \<open>[-M, M]\<close>, and the
  targets differ by at most \<open>eta\<close>, then the states differ by at most
  \<open>max \<bar>h 0 - g 0\<bar> (eta + 2 M k)\<close> at every \<open>t\<close>. The difference is itself a convex
  recurrence with weight \<open>a t\<close> and target \<open>w t\<close>, so the bound is
  @{text convex_recurrence_bounded} applied to it. The hypothesis \<open>0 \<le> k\<close> is needed only
  when \<open>a t = 1\<close> for every \<open>t\<close>; otherwise the gate hypothesis implies it.
\<close>

lemma cell_sensitivity_bound:
  fixes a b c d h g :: "nat \<Rightarrow> real" and k M \<eta> :: real
  assumes step_h: "\<And>t. h (Suc t) = a t * h t + (1 - a t) * c t"
    and step_g: "\<And>t. g (Suc t) = b t * g t + (1 - b t) * d t"
    and a_lo: "\<And>t. 0 \<le> a t" and a_hi: "\<And>t. a t \<le> 1"
    and k_nonneg: "0 \<le> k"
    and gate: "\<And>t. \<bar>b t - a t\<bar> \<le> k * (1 - a t)"
    and d_bd: "\<And>t. \<bar>d t\<bar> \<le> M"
    and g_bd: "\<And>t. \<bar>g t\<bar> \<le> M"
    and drift: "\<And>t. \<bar>c t - d t\<bar> \<le> \<eta>"
  shows "\<bar>h t - g t\<bar> \<le> max \<bar>h 0 - g 0\<bar> (\<eta> + 2 * M * k)"
proof -
  define w where "w \<equiv> \<lambda>t. (c t - d t)
      + (if a t = 1 then 0 else (a t - b t) / (1 - a t) * (g t - d t))"
  have e_step: "h (Suc t) - g (Suc t) = a t * (h t - g t) + (1 - a t) * w t" for t
  proof (cases "a t = 1")
    case True
    then have "b t = a t"
      using gate[of t] by simp
    then show ?thesis
      by (simp add: step_h step_g w_def True)
  next
    case False
    have nz: "1 - a t \<noteq> 0"
      using False by simp
    have rec: "h (Suc t) - g (Suc t)
        = a t * (h t - g t) + (1 - a t) * (c t - d t) + (a t - b t) * (g t - d t)"
      by (simp only: step_h step_g error_recursion)
    have cancel: "(1 - a t) * ((a t - b t) / (1 - a t) * (g t - d t))
        = (a t - b t) * (g t - d t)"
      using nz False by (simp add: field_simps)
    have wt: "w t = (c t - d t) + (a t - b t) / (1 - a t) * (g t - d t)"
      unfolding w_def using False by simp
    show ?thesis
      using rec cancel by (simp add: wt distrib_left)
  qed
  have M_nonneg: "0 \<le> M"
    using d_bd[of 0] by simp
  have w_bd: "\<bar>w t\<bar> \<le> \<eta> + 2 * M * k" for t
  proof (cases "a t = 1")
    case True
    have "\<bar>w t\<bar> = \<bar>c t - d t\<bar>"
      unfolding w_def using True by simp
    also have "\<dots> \<le> \<eta>"
      by (rule drift)
    also have "\<dots> \<le> \<eta> + 2 * M * k"
      using M_nonneg k_nonneg by simp
    finally show ?thesis .
  next
    case False
    have pos: "0 < 1 - a t"
      using False a_hi[of t] by simp
    have flip: "\<bar>a t - b t\<bar> = \<bar>b t - a t\<bar>"
      by (rule abs_minus_commute)
    have q: "\<bar>(a t - b t) / (1 - a t)\<bar> \<le> k"
      using gate[of t] pos flip by (simp add: abs_divide divide_le_eq)
    have gd: "\<bar>g t - d t\<bar> \<le> 2 * M"
      using abs_triangle_ineq4[of "g t" "d t"] g_bd[of t] d_bd[of t] by linarith
    have wt: "w t = (c t - d t) + (a t - b t) / (1 - a t) * (g t - d t)"
      unfolding w_def using False by simp
    have "\<bar>w t\<bar> \<le> \<bar>c t - d t\<bar> + \<bar>(a t - b t) / (1 - a t) * (g t - d t)\<bar>"
      unfolding wt by (rule abs_triangle_ineq)
    also have "\<dots> = \<bar>c t - d t\<bar> + \<bar>(a t - b t) / (1 - a t)\<bar> * \<bar>g t - d t\<bar>"
      by (simp only: abs_mult)
    also have "\<dots> \<le> \<eta> + k * (2 * M)"
      using drift[of t] q gd k_nonneg by (intro add_mono mult_mono) auto
    also have "\<dots> = \<eta> + 2 * M * k"
      by (simp add: algebra_simps)
    finally show ?thesis .
  qed
  show ?thesis
    by (rule convex_recurrence_bounded[where h = "\<lambda>t. h t - g t" and a = a and c = w
          and M = "\<eta> + 2 * M * k", OF e_step a_lo a_hi w_bd])
qed

end
