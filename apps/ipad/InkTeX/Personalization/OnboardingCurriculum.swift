import Combine
import Foundation

// MARK: - Math Categories Covered by Adaptive Onboarding

public enum OnboardingCategory: String, CaseIterable, Codable, Sendable, Identifiable {
    case digitsAndConfusables
    case algebraAndScripts
    case fractionsAndRoots
    case greekLetters
    case calculusAndBigOps
    case functionsAndTrig
    case relationsAndLogic
    case accentsAndVectors

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .digitsAndConfusables: return "Digits & Lookalikes"
        case .algebraAndScripts: return "Algebra & Scripts"
        case .fractionsAndRoots: return "Fractions & Roots"
        case .greekLetters: return "Greek & Symbols"
        case .calculusAndBigOps: return "Calculus & Sums"
        case .functionsAndTrig: return "Trig & Functions"
        case .relationsAndLogic: return "Relations & Sets"
        case .accentsAndVectors: return "Vectors & Grouping"
        }
    }

    public var shortTitle: String {
        switch self {
        case .digitsAndConfusables: return "Digits"
        case .algebraAndScripts: return "Algebra"
        case .fractionsAndRoots: return "Fractions"
        case .greekLetters: return "Greek"
        case .calculusAndBigOps: return "Calculus"
        case .functionsAndTrig: return "Functions"
        case .relationsAndLogic: return "Relations"
        case .accentsAndVectors: return "Vectors"
        }
    }

    public var iconName: String {
        switch self {
        case .digitsAndConfusables: return "textformat.123"
        case .algebraAndScripts: return "x.squareroot"
        case .fractionsAndRoots: return "divide"
        case .greekLetters: return "character"
        case .calculusAndBigOps: return "sum"
        case .functionsAndTrig: return "waveform.path.ecg"
        case .relationsAndLogic: return "equal.square"
        case .accentsAndVectors: return "arrow.up.right"
        }
    }
}

public enum OnboardingPreset: String, CaseIterable, Identifiable, Sendable {
    case quick
    case balanced
    case deep

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .quick: return "Quick"
        case .balanced: return "Full"
        case .deep: return "Deep"
        }
    }

    public var totalPrompts: Int {
        switch self {
        case .quick: return 12
        case .balanced: return 28
        case .deep: return 44
        }
    }

    /// Number of breadth-first diagnostic prompts before switching to weakness-targeted adaptive drills.
    public var breadthCount: Int {
        switch self {
        case .quick: return 8
        case .balanced: return 16
        case .deep: return 24
        }
    }

    public var subtitle: String {
        switch self {
        case .quick: return "12 eqs · ~2 min · Tier 1 FiLM + baseline"
        case .balanced: return "28 eqs · ~5 min · All tiers + adaptive drills"
        case .deep: return "44 eqs · ~8 min · Max Tier 2 LoRA accuracy"
        }
    }
}

public struct OnboardingPrompt: Identifiable, Equatable, Sendable {
    public let id: String
    public let latex: String
    public let category: OnboardingCategory
    public let focusSymbols: [String]
    public let difficulty: Int
    public let tip: String

    public init(
        id: String,
        latex: String,
        category: OnboardingCategory,
        focusSymbols: [String],
        difficulty: Int = 1,
        tip: String = "Write at your natural speed."
    ) {
        self.id = id
        self.latex = latex
        self.category = category
        self.focusSymbols = focusSymbols
        self.difficulty = difficulty
        self.tip = tip
    }
}

public struct SymbolConfusion: Equatable, Sendable, Identifiable {
    public var id: String { "\(expected)->\(predicted)" }
    public let expected: String
    public let predicted: String
}

public struct OnboardingEvaluation: Sendable {
    public let prompt: OnboardingPrompt
    public let predictedLatex: String
    public let canonicalTarget: String
    public let canonicalPredicted: String
    public let isExactMatch: Bool
    public let tokenAccuracy: Double
    public let confidence: Float
    public let latencyMs: Double
    public let missedSymbols: [String]
    public let confusions: [SymbolConfusion]
    public let strokes: [RawStroke]
    public let usedWriterStyle: Bool
    public let usedLoRA: Bool
    public let selectionReason: String
}

public struct CategoryMasterySummary: Identifiable, Sendable {
    public var id: String { category.id }
    public let category: OnboardingCategory
    public let attempted: Int
    public let exactMatches: Int
    public let avgTokenAccuracy: Double
    public let weakSymbols: [String]

    public var accuracy: Double {
        attempted > 0 ? Double(exactMatches) / Double(attempted) : 0
    }
}

// MARK: - Canonical LaTeX Tokenizer & Evaluator

public enum OnboardingEvaluator {
    private static let syntaxIgnoredForWeakness: Set<String> = [
        "{", "}", "(", ")", ",", ".", "=", "+", "-", " ",
    ]

    /// Tokenizes LaTeX into InkTeX-compatible tokens (`\command` or single non-space character).
    public static func tokenize(_ raw: String) -> [String] {
        var tokens: [String] = []
        let chars = Array(raw)
        var i = 0
        while i < chars.count {
            let ch = chars[i]
            if ch.isWhitespace {
                i += 1
                continue
            }
            if ch == "\\" {
                var j = i + 1
                if j < chars.count {
                    if chars[j].isLetter {
                        while j < chars.count && chars[j].isLetter {
                            j += 1
                        }
                        tokens.append(String(chars[i..<j]))
                        i = j
                        continue
                    } else {
                        tokens.append(String(chars[i...j]))
                        i = j + 1
                        continue
                    }
                }
            }
            tokens.append(String(ch))
            i += 1
        }
        return tokens
    }

    /// Canonicalizes token stream so semantically identical LaTeX (e.g. `x^{2}` vs `x^2`, `\le` vs `\leq`)
    /// is never falsely flagged as a recognition error during onboarding evaluation.
    public static func canonicalTokens(_ raw: String) -> [String] {
        let aliasMap: [String: String] = [
            "\\le": "\\leq",
            "\\ge": "\\geq",
            "\\ne": "\\neq",
            "\\to": "\\rightarrow",
            "\\dfrac": "\\frac",
            "\\tfrac": "\\frac",
            "\\cdotp": "\\cdot",
            "\\vert": "|",
            "\\lbrack": "[",
            "\\rbrack": "]",
            "\\varepsilon": "\\epsilon",
            "\\varphi": "\\phi",
            "\\vartheta": "\\theta",
        ]
        let dropTokens: Set<String> = [
            "\\left", "\\right", "\\,", "\\;", "\\:", "\\!", "\\ ", "\\quad", "\\qquad",
            "\\limits", "\\nolimits", "\\displaystyle", "\\textstyle",
        ]

        var rawToks: [String] = []
        for t in tokenize(raw) {
            if dropTokens.contains(t) { continue }
            rawToks.append(aliasMap[t] ?? t)
        }

        // Collapse single-token brace groups after `^` or `_`: `^ { 2 }` -> `^ 2`
        var out: [String] = []
        var i = 0
        while i < rawToks.count {
            if (rawToks[i] == "^" || rawToks[i] == "_"),
               i + 3 < rawToks.count,
               rawToks[i + 1] == "{",
               rawToks[i + 3] == "}",
               rawToks[i + 2] != "{" && rawToks[i + 2] != "}" {
                out.append(rawToks[i])
                out.append(rawToks[i + 2])
                i += 4
                continue
            }
            out.append(rawToks[i])
            i += 1
        }
        return out
    }

    public static func evaluate(
        prompt: OnboardingPrompt,
        result: RecognitionResult,
        strokes: [RawStroke],
        selectionReason: String
    ) -> OnboardingEvaluation {
        let tgt = canonicalTokens(prompt.latex)
        let pred = canonicalTokens(result.latex)
        let canonTgt = LatexTokenizer.detokenize(tgt)
        let canonPred = LatexTokenizer.detokenize(pred)
        let exact = (tgt == pred)

        let (dist, missed, confusions) = alignAndDiff(target: tgt, predicted: pred)
        let denom = max(1, max(tgt.count, pred.count))
        let tokAcc = max(0.0, 1.0 - Double(dist) / Double(denom))

        return OnboardingEvaluation(
            prompt: prompt,
            predictedLatex: result.latex,
            canonicalTarget: canonTgt,
            canonicalPredicted: canonPred,
            isExactMatch: exact,
            tokenAccuracy: exact ? 1.0 : tokAcc,
            confidence: result.overallConfidence,
            latencyMs: result.totalMs,
            missedSymbols: missed,
            confusions: confusions,
            strokes: strokes,
            usedWriterStyle: result.usedWriterStyle,
            usedLoRA: result.usedLoRAAdapter,
            selectionReason: selectionReason
        )
    }

    /// Levenshtein DP + backtrace to pinpoint exact missed tokens and substitution pairs.
    private static func alignAndDiff(
        target: [String],
        predicted: [String]
    ) -> (distance: Int, missed: [String], confusions: [SymbolConfusion]) {
        let m = target.count
        let n = predicted.count
        if m == 0 { return (n, [], []) }
        if n == 0 {
            var seen: Set<String> = []
            let missed = target.filter { !syntaxIgnoredForWeakness.contains($0) && seen.insert($0).inserted }
            return (m, missed, [])
        }

        var dp = Array(repeating: Array(repeating: 0, count: n + 1), count: m + 1)
        for i in 0...m { dp[i][0] = i }
        for j in 0...n { dp[0][j] = j }

        for i in 1...m {
            for j in 1...n {
                if target[i - 1] == predicted[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1]
                } else {
                    dp[i][j] = 1 + min(dp[i - 1][j - 1], min(dp[i - 1][j], dp[i][j - 1]))
                }
            }
        }

        var missed: [String] = []
        var confusions: [SymbolConfusion] = []
        var i = m
        var j = n
        while i > 0 || j > 0 {
            if i > 0 && j > 0 && target[i - 1] == predicted[j - 1] {
                i -= 1
                j -= 1
            } else if i > 0 && j > 0 && dp[i][j] == dp[i - 1][j - 1] + 1 {
                let exp = target[i - 1]
                let got = predicted[j - 1]
                if !syntaxIgnoredForWeakness.contains(exp) {
                    missed.append(exp)
                    confusions.append(SymbolConfusion(expected: exp, predicted: got))
                }
                i -= 1
                j -= 1
            } else if i > 0 && dp[i][j] == dp[i - 1][j] + 1 {
                let exp = target[i - 1]
                if !syntaxIgnoredForWeakness.contains(exp) {
                    missed.append(exp)
                }
                i -= 1
            } else {
                j -= 1
            }
        }

        var seenMissed: Set<String> = []
        let uniqueMissed = missed.reversed().filter { seenMissed.insert($0).inserted }
        var seenConf: Set<String> = []
        let uniqueConf = confusions.reversed().filter { seenConf.insert($0.id).inserted }
        return (dp[m][n], uniqueMissed, uniqueConf)
    }
}

// MARK: - Comprehensive Canonical Prompt Bank (56 Prompts across 8 Categories)

public enum OnboardingCurriculum {
    /// All 56 prompts strictly use canonical InkTeX v2 tokens (0 `<UNK>` tokens, 100% LoRA-trainable).
    public static let promptBank: [OnboardingPrompt] = [
        // 1. Digits & Lookalikes (catches 2/z, 5/s, 1/l/7, 0/O, 9/q/g, 4/9, 6/b, +/t, x/\times, u/v)
        OnboardingPrompt(
            id: "dig_01",
            latex: "2z + 5s = 17",
            category: .digitsAndConfusables,
            focusSymbols: ["2", "z", "5", "s", "1", "7"],
            difficulty: 1,
            tip: "Write 2 vs z and 5 vs s the way you naturally do."
        ),
        OnboardingPrompt(
            id: "dig_02",
            latex: "9q - 4g = 60",
            category: .digitsAndConfusables,
            focusSymbols: ["9", "q", "4", "g", "6", "0"],
            difficulty: 1,
            tip: "Helps InkTeX learn your loops on 9, q, g, and 6."
        ),
        OnboardingPrompt(
            id: "dig_03",
            latex: "1 + l = 7 - t",
            category: .digitsAndConfusables,
            focusSymbols: ["1", "+", "l", "7", "t"],
            difficulty: 2,
            tip: "Distinguishes your 1, lowercase l, 7, and t."
        ),
        OnboardingPrompt(
            id: "dig_04",
            latex: "3x \\times 8y = 24",
            category: .digitsAndConfusables,
            focusSymbols: ["3", "x", "\\times", "8", "y", "2", "4"],
            difficulty: 1,
            tip: "Captures how you separate variable x from multiplication ×."
        ),
        OnboardingPrompt(
            id: "dig_05",
            latex: "b + 6 = 0.5s",
            category: .digitsAndConfusables,
            focusSymbols: ["b", "6", "0", ".", "5", "s"],
            difficulty: 2,
            tip: "Calibrates b vs 6, decimal points, and 5 vs s."
        ),
        OnboardingPrompt(
            id: "dig_06",
            latex: "u + v = 9n + h",
            category: .digitsAndConfusables,
            focusSymbols: ["u", "v", "9", "n", "h"],
            difficulty: 2,
            tip: "Distinguishes curved letters u, v, n, and h."
        ),
        OnboardingPrompt(
            id: "dig_07",
            latex: "z^2 - 2z + 5 = s",
            category: .digitsAndConfusables,
            focusSymbols: ["z", "2", "^", "5", "s"],
            difficulty: 3,
            tip: "Combines z, 2, 5, and s with an exponent."
        ),

        // 2. Algebra, Powers & Subscripts
        OnboardingPrompt(
            id: "alg_01",
            latex: "x^2 + y^2 = r^2",
            category: .algebraAndScripts,
            focusSymbols: ["x", "y", "r", "^", "2"],
            difficulty: 1,
            tip: "Classic baseline equation for exponent height and spacing."
        ),
        OnboardingPrompt(
            id: "alg_02",
            latex: "a_1 + a_2 = a_n",
            category: .algebraAndScripts,
            focusSymbols: ["a", "_", "1", "2", "n"],
            difficulty: 1,
            tip: "Calibrates your subscript position relative to the baseline."
        ),
        OnboardingPrompt(
            id: "alg_03",
            latex: "f'(x) = 3x^2 - 2x + 1",
            category: .algebraAndScripts,
            focusSymbols: ["f", "'", "x", "^", "3", "2"],
            difficulty: 2,
            tip: "Captures primes, parentheses, and polynomial terms."
        ),
        OnboardingPrompt(
            id: "alg_04",
            latex: "(a + b)^2 = a^2 + 2ab + b^2",
            category: .algebraAndScripts,
            focusSymbols: ["(", ")", "a", "b", "^", "2"],
            difficulty: 2,
            tip: "Tests longer horizontal spacing and parenthesized powers."
        ),
        OnboardingPrompt(
            id: "alg_05",
            latex: "x_{i+1} = 2x_i - 1",
            category: .algebraAndScripts,
            focusSymbols: ["x", "_", "i", "1", "2"],
            difficulty: 2,
            tip: "Multi-character subscript grouping."
        ),
        OnboardingPrompt(
            id: "alg_06",
            latex: "e^{-x^2} + k_n",
            category: .algebraAndScripts,
            focusSymbols: ["e", "^", "x", "2", "k", "_", "n"],
            difficulty: 3,
            tip: "Nested exponent plus subscript."
        ),
        OnboardingPrompt(
            id: "alg_07",
            latex: "p(x) = c_0 + c_1 x + c_2 x^2",
            category: .algebraAndScripts,
            focusSymbols: ["p", "c", "_", "0", "1", "2", "x", "^"],
            difficulty: 3,
            tip: "Tests mixing subscripts and superscripts in one line."
        ),

        // 3. Fractions & Radicals
        OnboardingPrompt(
            id: "frc_01",
            latex: "\\frac{a+b}{c}",
            category: .fractionsAndRoots,
            focusSymbols: ["\\frac", "a", "b", "c"],
            difficulty: 1,
            tip: "Captures your fraction bar stroke length and order."
        ),
        OnboardingPrompt(
            id: "frc_02",
            latex: "\\sqrt{x^2 + y^2}",
            category: .fractionsAndRoots,
            focusSymbols: ["\\sqrt", "x", "y", "^", "2"],
            difficulty: 1,
            tip: "Calibrates how you draw radical roofs and enclosed powers."
        ),
        OnboardingPrompt(
            id: "frc_03",
            latex: "\\frac{1}{x} + \\frac{1}{y} = \\frac{x+y}{xy}",
            category: .fractionsAndRoots,
            focusSymbols: ["\\frac", "1", "x", "y"],
            difficulty: 2,
            tip: "Multiple side-by-side fractions."
        ),
        OnboardingPrompt(
            id: "frc_04",
            latex: "\\frac{1}{\\sqrt{2\\pi}}",
            category: .fractionsAndRoots,
            focusSymbols: ["\\frac", "\\sqrt", "1", "2", "\\pi"],
            difficulty: 2,
            tip: "Radical nested inside a denominator."
        ),
        OnboardingPrompt(
            id: "frc_05",
            latex: "x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}",
            category: .fractionsAndRoots,
            focusSymbols: ["\\frac", "\\pm", "\\sqrt", "b", "4", "a", "c"],
            difficulty: 3,
            tip: "Quadratic formula: tests long fraction bar + ± + radical."
        ),
        OnboardingPrompt(
            id: "frc_06",
            latex: "\\sqrt[3]{x + 1} = 2",
            category: .fractionsAndRoots,
            focusSymbols: ["\\sqrt", "[", "3", "]", "x", "1", "2"],
            difficulty: 3,
            tip: "Cube root index tucked into the radical hook."
        ),
        OnboardingPrompt(
            id: "frc_07",
            latex: "\\frac{a/b}{c} = \\frac{a}{bc}",
            category: .fractionsAndRoots,
            focusSymbols: ["\\frac", "/", "a", "b", "c"],
            difficulty: 3,
            tip: "Inline slash vs horizontal fraction bar."
        ),

        // 4. Greek Letters & Special Symbols
        OnboardingPrompt(
            id: "grk_01",
            latex: "\\alpha + \\beta = \\gamma",
            category: .greekLetters,
            focusSymbols: ["\\alpha", "\\beta", "\\gamma"],
            difficulty: 1,
            tip: "Core Greek strokes — α, β, and γ often vary a lot by writer."
        ),
        OnboardingPrompt(
            id: "grk_02",
            latex: "\\theta = \\frac{\\pi}{4}",
            category: .greekLetters,
            focusSymbols: ["\\theta", "\\pi", "\\frac", "4"],
            difficulty: 1,
            tip: "Calibrates your θ crossbar and π top stroke."
        ),
        OnboardingPrompt(
            id: "grk_03",
            latex: "\\Delta x = \\lambda \\mu",
            category: .greekLetters,
            focusSymbols: ["\\Delta", "x", "\\lambda", "\\mu"],
            difficulty: 2,
            tip: "Triangle Δ, λ slant, and μ tail."
        ),
        OnboardingPrompt(
            id: "grk_04",
            latex: "\\sigma^2 + \\tau = \\omega",
            category: .greekLetters,
            focusSymbols: ["\\sigma", "^", "2", "\\tau", "\\omega"],
            difficulty: 2,
            tip: "Helps separate σ from 6/0 and ω from w."
        ),
        OnboardingPrompt(
            id: "grk_05",
            latex: "\\phi(x) = \\psi(x) + \\epsilon",
            category: .greekLetters,
            focusSymbols: ["\\phi", "\\psi", "\\epsilon", "x"],
            difficulty: 2,
            tip: "Covers φ, ψ, and ε."
        ),
        OnboardingPrompt(
            id: "grk_06",
            latex: "\\delta \\rho = \\eta \\xi",
            category: .greekLetters,
            focusSymbols: ["\\delta", "\\rho", "\\eta", "\\xi"],
            difficulty: 3,
            tip: "Distinctive Greek shapes: δ, ρ, η, and ξ."
        ),
        OnboardingPrompt(
            id: "grk_07",
            latex: "\\Omega = \\Sigma + \\Phi \\infty",
            category: .greekLetters,
            focusSymbols: ["\\Omega", "\\Sigma", "\\Phi", "\\infty"],
            difficulty: 3,
            tip: "Uppercase Greek letters and infinity ∞."
        ),

        // 5. Calculus, Sums & Limits
        OnboardingPrompt(
            id: "cal_01",
            latex: "\\int_0^1 x^2 dx",
            category: .calculusAndBigOps,
            focusSymbols: ["\\int", "_", "0", "^", "1", "x", "d"],
            difficulty: 1,
            tip: "Captures your integral curve and bound placement."
        ),
        OnboardingPrompt(
            id: "cal_02",
            latex: "\\sum_{i=1}^n i = \\frac{n(n+1)}{2}",
            category: .calculusAndBigOps,
            focusSymbols: ["\\sum", "_", "^", "i", "n", "\\frac"],
            difficulty: 1,
            tip: "Summation symbol with lower and upper limits."
        ),
        OnboardingPrompt(
            id: "cal_03",
            latex: "\\lim_{x \\rightarrow 0} \\frac{\\sin x}{x} = 1",
            category: .calculusAndBigOps,
            focusSymbols: ["\\lim", "\\rightarrow", "\\sin", "\\frac", "x", "0"],
            difficulty: 2,
            tip: "Limit subscript with right arrow →."
        ),
        OnboardingPrompt(
            id: "cal_04",
            latex: "\\frac{dy}{dx} = 2x",
            category: .calculusAndBigOps,
            focusSymbols: ["\\frac", "d", "y", "x", "2"],
            difficulty: 1,
            tip: "Leibniz derivative notation."
        ),
        OnboardingPrompt(
            id: "cal_05",
            latex: "\\frac{\\partial f}{\\partial x} + \\frac{\\partial f}{\\partial y}",
            category: .calculusAndBigOps,
            focusSymbols: ["\\partial", "\\frac", "f", "x", "y"],
            difficulty: 2,
            tip: "Partial derivative ∂ loops."
        ),
        OnboardingPrompt(
            id: "cal_06",
            latex: "\\int_0^\\infty e^{-t} dt = 1",
            category: .calculusAndBigOps,
            focusSymbols: ["\\int", "\\infty", "e", "^", "t", "d"],
            difficulty: 3,
            tip: "Improper integral with ∞ and negative exponent."
        ),
        OnboardingPrompt(
            id: "cal_07",
            latex: "\\prod_{k=1}^n k = n!",
            category: .calculusAndBigOps,
            focusSymbols: ["\\prod", "_", "^", "k", "n", "!"],
            difficulty: 3,
            tip: "Big product Π with bounds and factorial."
        ),

        // 6. Functions, Trig & Logs
        OnboardingPrompt(
            id: "fun_01",
            latex: "\\sin^2 \\theta + \\cos^2 \\theta = 1",
            category: .functionsAndTrig,
            focusSymbols: ["\\sin", "\\cos", "^", "2", "\\theta", "1"],
            difficulty: 1,
            tip: "Trig identity with exponents on function names."
        ),
        OnboardingPrompt(
            id: "fun_02",
            latex: "\\ln(ab) = \\ln a + \\ln b",
            category: .functionsAndTrig,
            focusSymbols: ["\\ln", "a", "b"],
            difficulty: 1,
            tip: "Natural log handwriting (often confused with 1n)."
        ),
        OnboardingPrompt(
            id: "fun_03",
            latex: "\\tan x = \\frac{\\sin x}{\\cos x}",
            category: .functionsAndTrig,
            focusSymbols: ["\\tan", "\\sin", "\\cos", "\\frac", "x"],
            difficulty: 2,
            tip: "Tests sin, cos, and tan together."
        ),
        OnboardingPrompt(
            id: "fun_04",
            latex: "\\log_2 8 = 3",
            category: .functionsAndTrig,
            focusSymbols: ["\\log", "_", "2", "8", "3"],
            difficulty: 2,
            tip: "Logarithm with base subscript."
        ),
        OnboardingPrompt(
            id: "fun_05",
            latex: "\\exp(i\\pi) + 1 = 0",
            category: .functionsAndTrig,
            focusSymbols: ["\\exp", "i", "\\pi", "1", "0"],
            difficulty: 2,
            tip: "Named exp() function with imaginary i and π."
        ),
        OnboardingPrompt(
            id: "fun_06",
            latex: "\\max(x, y) - \\min(x, y)",
            category: .functionsAndTrig,
            focusSymbols: ["\\max", "\\min", "x", "y"],
            difficulty: 3,
            tip: "Multi-letter max/min operator names."
        ),
        OnboardingPrompt(
            id: "fun_07",
            latex: "\\det(A) = ad - bc",
            category: .functionsAndTrig,
            focusSymbols: ["\\det", "A", "a", "d", "b", "c"],
            difficulty: 3,
            tip: "Determinant operator and uppercase A."
        ),

        // 7. Relations, Sets & Logic
        OnboardingPrompt(
            id: "rel_01",
            latex: "x \\leq y \\neq z",
            category: .relationsAndLogic,
            focusSymbols: ["\\leq", "\\neq", "x", "y", "z"],
            difficulty: 1,
            tip: "Inequality ≤ and not-equal ≠ slash."
        ),
        OnboardingPrompt(
            id: "rel_02",
            latex: "a \\geq b \\approx c",
            category: .relationsAndLogic,
            focusSymbols: ["\\geq", "\\approx", "a", "b", "c"],
            difficulty: 1,
            tip: "Greater-or-equal ≥ and wavy ≈."
        ),
        OnboardingPrompt(
            id: "rel_03",
            latex: "x \\in A \\cup B",
            category: .relationsAndLogic,
            focusSymbols: ["\\in", "\\cup", "x", "A", "B"],
            difficulty: 2,
            tip: "Set membership ∈ vs ε and union ∪."
        ),
        OnboardingPrompt(
            id: "rel_04",
            latex: "A \\cap B \\subset C",
            category: .relationsAndLogic,
            focusSymbols: ["\\cap", "\\subset", "A", "B", "C"],
            difficulty: 2,
            tip: "Intersection ∩ and subset ⊂."
        ),
        OnboardingPrompt(
            id: "rel_05",
            latex: "\\forall x \\in \\mathbb{R}, x^2 \\geq 0",
            category: .relationsAndLogic,
            focusSymbols: ["\\forall", "\\in", "\\mathbb", "R", "\\geq", "0"],
            difficulty: 3,
            tip: "Universal quantifier ∀ and blackboard bold ℝ."
        ),
        OnboardingPrompt(
            id: "rel_06",
            latex: "\\exists n > 0 \\Rightarrow n \\geq 1",
            category: .relationsAndLogic,
            focusSymbols: ["\\exists", ">", "\\Rightarrow", "\\geq", "n", "1"],
            difficulty: 3,
            tip: "Existential ∃ and double implication arrow ⇒."
        ),
        OnboardingPrompt(
            id: "rel_07",
            latex: "A \\subseteq B \\Leftrightarrow x \\notin \\emptyset",
            category: .relationsAndLogic,
            focusSymbols: ["\\subseteq", "\\Leftrightarrow", "\\notin", "\\emptyset"],
            difficulty: 3,
            tip: "Covers ⊆, ⇔, ∉, and empty set ∅."
        ),

        // 8. Vectors, Accents & Grouping
        OnboardingPrompt(
            id: "vec_01",
            latex: "|x - y| \\leq |x| + |y|",
            category: .accentsAndVectors,
            focusSymbols: ["|", "\\leq", "x", "y"],
            difficulty: 1,
            tip: "Vertical bars | vs 1 or lowercase l."
        ),
        OnboardingPrompt(
            id: "vec_02",
            latex: "\\vec{u} \\cdot \\vec{v} = 0",
            category: .accentsAndVectors,
            focusSymbols: ["\\vec", "\\cdot", "u", "v", "0"],
            difficulty: 1,
            tip: "Vector arrows above letters and dot product."
        ),
        OnboardingPrompt(
            id: "vec_03",
            latex: "\\hat{y} = \\overline{x} + \\beta",
            category: .accentsAndVectors,
            focusSymbols: ["\\hat", "\\overline", "y", "x", "\\beta"],
            difficulty: 2,
            tip: "Hat accent and overline bar."
        ),
        OnboardingPrompt(
            id: "vec_04",
            latex: "S = \\{1, 2, 3\\}",
            category: .accentsAndVectors,
            focusSymbols: ["S", "\\{", "\\}", "1", "2", "3"],
            difficulty: 2,
            tip: "Curly set braces { }."
        ),
        OnboardingPrompt(
            id: "vec_05",
            latex: "[a, b] \\cap (c, d)",
            category: .accentsAndVectors,
            focusSymbols: ["[", "]", "\\cap", "a", "b", "c", "d"],
            difficulty: 2,
            tip: "Square brackets vs parentheses."
        ),
        OnboardingPrompt(
            id: "vec_06",
            latex: "\\nabla f = (\\frac{\\partial f}{\\partial x}, \\frac{\\partial f}{\\partial y})",
            category: .accentsAndVectors,
            focusSymbols: ["\\nabla", "\\partial", "\\frac", "f", "x", "y"],
            difficulty: 3,
            tip: "Gradient operator ∇ with partial vector."
        ),
        OnboardingPrompt(
            id: "vec_07",
            latex: "\\tilde{x} + \\dot{y} = \\angle ABC",
            category: .accentsAndVectors,
            focusSymbols: ["\\tilde", "\\dot", "\\angle", "x", "y", "A", "B", "C"],
            difficulty: 3,
            tip: "Tilde, time-derivative dot, and angle symbol."
        ),
    ]
}

// MARK: - Adaptive Onboarding Session Controller

@MainActor
public final class AdaptiveOnboardingSession: ObservableObject {
    public enum Phase: Equatable {
        case welcome
        case writing
        case evaluating
        case feedback
        case summary
    }

    @Published public var phase: Phase = .welcome
    @Published public var preset: OnboardingPreset = .balanced
    @Published public var targetTotal: Int = OnboardingPreset.balanced.totalPrompts
    @Published public var breadthTarget: Int = OnboardingPreset.balanced.breadthCount
    @Published public private(set) var currentPrompt: OnboardingPrompt?
    @Published public private(set) var currentSelectionReason: String = ""
    @Published public private(set) var isAdaptiveDrillPhase: Bool = false
    @Published public private(set) var evaluations: [OnboardingEvaluation] = []
    @Published public private(set) var latestEvaluation: OnboardingEvaluation?
    @Published public private(set) var totalStrokesCollected: Int = 0
    @Published public private(set) var styleEnrolledInSession: Bool = false
    @Published public private(set) var styleEnrollmentBanner: String?
    @Published public private(set) var isUpdatingStyle: Bool = false
    @Published public private(set) var evaluationError: String?

    /// Running symbol miss counts (token -> times missed or misread).
    private var missedSymbolCounts: [String: Int] = [:]
    /// Running per-category attempts, errors, and sum of confidences.
    private var categoryAttempts: [OnboardingCategory: Int] = [:]
    private var categoryErrors: [OnboardingCategory: Int] = [:]
    private var categoryConfidenceSum: [OnboardingCategory: Double] = [:]
    private var usedPromptIds: Set<String> = []
    private var skippedPromptIds: Set<String> = []
    /// Diverse calibration stroke pool gathered across equations for Tier 1 FiLM.
    private var calibrationStrokePool: [RawStroke] = []
    private var lastStyleEnrollStrokeCount: Int = 0
    private var breadthQueue: [OnboardingPrompt] = []

    public init() {}

    public var completedCount: Int { evaluations.count }

    public var exactMatchCount: Int { evaluations.filter(\.isExactMatch).count }

    public var exactMatchRate: Double {
        evaluations.isEmpty ? 0 : Double(exactMatchCount) / Double(evaluations.count)
    }

    public var averageTokenAccuracy: Double {
        guard !evaluations.isEmpty else { return 0 }
        return evaluations.reduce(0.0) { $0 + $1.tokenAccuracy } / Double(evaluations.count)
    }

    public var topWeakSymbols: [(symbol: String, misses: Int)] {
        missedSymbolCounts
            .sorted { lhs, rhs in
                if lhs.value != rhs.value { return lhs.value > rhs.value }
                return lhs.key < rhs.key
            }
            .prefix(8)
            .map { ($0.key, $0.value) }
    }

    public var categorySummaries: [CategoryMasterySummary] {
        OnboardingCategory.allCases.map { cat in
            let catEvals = evaluations.filter { $0.prompt.category == cat }
            let exact = catEvals.filter(\.isExactMatch).count
            let avgTok = catEvals.isEmpty
                ? 0
                : catEvals.reduce(0.0) { $0 + $1.tokenAccuracy } / Double(catEvals.count)
            var symCounts: [String: Int] = [:]
            for ev in catEvals {
                for sym in ev.missedSymbols {
                    symCounts[sym, default: 0] += 1
                }
            }
            let weak = symCounts.sorted { $0.value > $1.value }.prefix(4).map(\.key)
            return CategoryMasterySummary(
                category: cat,
                attempted: catEvals.count,
                exactMatches: exact,
                avgTokenAccuracy: avgTok,
                weakSymbols: weak
            )
        }
    }

    // MARK: - Session Control

    public func start(preset: OnboardingPreset, existingCalibrationStrokes: [RawStroke] = []) {
        self.preset = preset
        self.targetTotal = preset.totalPrompts
        self.breadthTarget = preset.breadthCount
        self.evaluations = []
        self.latestEvaluation = nil
        self.missedSymbolCounts = [:]
        self.categoryAttempts = [:]
        self.categoryErrors = [:]
        self.categoryConfidenceSum = [:]
        self.usedPromptIds = []
        self.skippedPromptIds = []
        self.calibrationStrokePool = Array(existingCalibrationStrokes.suffix(InkPreprocessorV2.styleMaxStrokes))
        self.totalStrokesCollected = 0
        self.styleEnrolledInSession = false
        self.styleEnrollmentBanner = nil
        self.lastStyleEnrollStrokeCount = 0
        self.evaluationError = nil

        buildBreadthQueue(count: preset.breadthCount)
        advanceToNextPrompt()
    }

    /// Extends a finished session with extra targeted drills on the user's weakest symbols & categories.
    public func extendWithTargetedDrills(extraCount: Int = 8) {
        let remainingBank = OnboardingCurriculum.promptBank.filter { !usedPromptIds.contains($0.id) }
        if remainingBank.isEmpty {
            // Allow repeating missed prompts if the entire 56-prompt bank was already exhausted
            usedPromptIds.removeAll()
        }
        targetTotal = evaluations.count + extraCount
        breadthQueue = []
        isAdaptiveDrillPhase = true
        advanceToNextPrompt()
    }

    public func skipCurrentPrompt() {
        if let cur = currentPrompt {
            skippedPromptIds.insert(cur.id)
            usedPromptIds.insert(cur.id)
        }
        latestEvaluation = nil
        evaluationError = nil
        advanceToNextPrompt()
    }

    public func retryCurrentPrompt() {
        if let last = evaluations.last, last.prompt.id == currentPrompt?.id {
            evaluations.removeLast()
            totalStrokesCollected = max(0, totalStrokesCollected - last.strokes.count)
            rebuildRunningStats()
        }
        latestEvaluation = nil
        evaluationError = nil
        phase = .writing
    }

    private func rebuildRunningStats() {
        missedSymbolCounts = [:]
        categoryAttempts = [:]
        categoryErrors = [:]
        categoryConfidenceSum = [:]
        for ev in evaluations {
            let cat = ev.prompt.category
            categoryAttempts[cat, default: 0] += 1
            if !ev.isExactMatch {
                categoryErrors[cat, default: 0] += 1
            }
            categoryConfidenceSum[cat, default: 0.0] += Double(ev.confidence)
            for sym in ev.missedSymbols {
                missedSymbolCounts[sym, default: 0] += 1
            }
            if ev.isExactMatch && ev.confidence < 0.75 {
                for sym in ev.prompt.focusSymbols.prefix(2) {
                    missedSymbolCounts[sym, default: 0] += 1
                }
            }
        }
    }

    public func finishEarly() {
        latestEvaluation = nil
        evaluationError = nil
        phase = .summary
    }

    /// Records an evaluation result, updates weakness weights, and returns strokes ready for Tier 1 FiLM if due.
    public func recordEvaluation(_ eval: OnboardingEvaluation, saveToPool: Bool = true) -> [RawStroke]? {
        latestEvaluation = eval
        evaluations.append(eval)
        usedPromptIds.insert(eval.prompt.id)
        totalStrokesCollected += eval.strokes.count

        let cat = eval.prompt.category
        categoryAttempts[cat, default: 0] += 1
        if !eval.isExactMatch {
            categoryErrors[cat, default: 0] += 1
        }
        categoryConfidenceSum[cat, default: 0.0] += Double(eval.confidence)

        for sym in eval.missedSymbols {
            missedSymbolCounts[sym, default: 0] += 1
        }
        // Also softly prioritize symbols from low-confidence matches (< 75%)
        if eval.isExactMatch && eval.confidence < 0.75 {
            for sym in eval.prompt.focusSymbols.prefix(2) {
                missedSymbolCounts[sym, default: 0] += 1
            }
        }

        phase = .feedback

        guard saveToPool else { return nil }

        // Keep a diverse sliding pool of strokes across equations for Tier 1 WriterStyleEncoder (max 20 strokes).
        // Take up to 5 strokes per equation so the 20-stroke FiLM vector represents multiple symbols & categories!
        let sampleFromEq = evenlySubsample(eval.strokes, maxCount: 5)
        calibrationStrokePool.append(contentsOf: sampleFromEq)
        if calibrationStrokePool.count > InkPreprocessorV2.styleMaxStrokes {
            calibrationStrokePool = evenlySubsample(calibrationStrokePool, maxCount: InkPreprocessorV2.styleMaxStrokes)
        }

        // Trigger Tier 1 FiLM enrollment as soon as we hit >= 12 strokes, then refresh once full (20 strokes)
        // and again halfway through the session so FiLM sees maximum symbol diversity.
        let poolCount = calibrationStrokePool.count
        let shouldEnrollNow =
            (poolCount >= 12 && !styleEnrolledInSession)
            || (poolCount >= InkPreprocessorV2.styleMaxStrokes && lastStyleEnrollStrokeCount < InkPreprocessorV2.styleMaxStrokes)
            || (poolCount >= InkPreprocessorV2.styleMaxStrokes && evaluations.count % 8 == 0)

        if shouldEnrollNow {
            lastStyleEnrollStrokeCount = poolCount
            return calibrationStrokePool
        }
        return nil
    }

    public func markStyleEnrolled(strokeCount: Int, elapsedMs: Double) {
        styleEnrolledInSession = true
        styleEnrollmentBanner = String(
            format: "Tier 1 Writer Style active (%d strokes · %.1f ms)",
            min(strokeCount, InkPreprocessorV2.styleMaxStrokes),
            elapsedMs
        )
    }

    public func proceedAfterFeedback() {
        latestEvaluation = nil
        evaluationError = nil
        styleEnrollmentBanner = nil
        if evaluations.count >= targetTotal {
            phase = .summary
        } else {
            advanceToNextPrompt()
        }
    }

    // MARK: - Adaptive Selection Logic

    private func buildBreadthQueue(count: Int) {
        // Round-robin across all 8 categories, ordered by difficulty 1 -> 2 -> 3
        // so the first 8 equations guarantee 100% category breadth!
        var byCat: [OnboardingCategory: [OnboardingPrompt]] = [:]
        for cat in OnboardingCategory.allCases {
            byCat[cat] = OnboardingCurriculum.promptBank
                .filter { $0.category == cat }
                .sorted { $0.difficulty < $1.difficulty }
        }
        var queue: [OnboardingPrompt] = []
        var round = 0
        while queue.count < count {
            var addedThisRound = false
            for cat in OnboardingCategory.allCases {
                if queue.count >= count { break }
                if let list = byCat[cat], round < list.count {
                    queue.append(list[round])
                    addedThisRound = true
                }
            }
            if !addedThisRound { break }
            round += 1
        }
        breadthQueue = queue
    }

    private func advanceToNextPrompt() {
        // 1. Breadth sweep first (guarantees full coverage of all 8 categories)
        while !breadthQueue.isEmpty {
            let next = breadthQueue.removeFirst()
            if usedPromptIds.contains(next.id) { continue }
            currentPrompt = next
            isAdaptiveDrillPhase = false
            let roundNum = (categoryAttempts[next.category] ?? 0) + 1
            currentSelectionReason = roundNum == 1
                ? "Coverage · \(next.category.shortTitle)"
                : "Variety · \(next.category.shortTitle)"
            phase = .writing
            return
        }

        // 2. Adaptive weakness-targeted drill selection
        isAdaptiveDrillPhase = true
        var candidates = OnboardingCurriculum.promptBank.filter {
            !usedPromptIds.contains($0.id) && !skippedPromptIds.contains($0.id)
        }
        if candidates.isEmpty {
            // If every single prompt was used, recycle prompts the user previously missed!
            let missedIds = Set(evaluations.filter { !$0.isExactMatch }.map { $0.prompt.id })
            candidates = OnboardingCurriculum.promptBank.filter { missedIds.contains($0.id) }
            if candidates.isEmpty {
                candidates = OnboardingCurriculum.promptBank
            }
        }

        let maxAttempts = max(1, categoryAttempts.values.max() ?? 1)
        var bestPrompt = candidates[0]
        var bestScore = -Double.greatestFiniteMagnitude
        var bestReason = "Adaptive Drill"

        for prompt in candidates {
            let matchedWeak = prompt.focusSymbols.filter { (missedSymbolCounts[$0] ?? 0) > 0 }
            let weakSymScore = matchedWeak.reduce(0.0) { acc, sym in
                acc + 4.5 * Double(missedSymbolCounts[sym] ?? 0)
            }

            let attempts = categoryAttempts[prompt.category] ?? 0
            let errors = categoryErrors[prompt.category] ?? 0
            let errRate = attempts > 0 ? Double(errors) / Double(attempts) : 0.35
            let avgConf = attempts > 0 ? (categoryConfidenceSum[prompt.category] ?? 0) / Double(attempts) : 0.85

            let catErrorScore = errRate * 5.0 + Double(errors) * 1.2
            let lowConfScore = max(0.0, (0.85 - avgConf)) * 4.0
            let diversityBonus = Double(maxAttempts - attempts) * 1.15
            let difficultyBonus = Double(prompt.difficulty) * 0.25

            let total = weakSymScore + catErrorScore + lowConfScore + diversityBonus + difficultyBonus
            if total > bestScore {
                bestScore = total
                bestPrompt = prompt
                if !matchedWeak.isEmpty {
                    let preview = matchedWeak.prefix(3).joined(separator: ", ")
                    bestReason = "Targeting missed: \(preview)"
                } else if errors > 0 {
                    bestReason = "Reinforcing \(prompt.category.shortTitle)"
                } else if avgConf < 0.82 {
                    bestReason = "Boosting confidence · \(prompt.category.shortTitle)"
                } else {
                    bestReason = "Expanding variety · \(prompt.category.shortTitle)"
                }
            }
        }

        currentPrompt = bestPrompt
        currentSelectionReason = bestReason
        phase = .writing
    }

    private func evenlySubsample(_ strokes: [RawStroke], maxCount: Int) -> [RawStroke] {
        guard strokes.count > maxCount, maxCount > 0 else { return strokes }
        var out: [RawStroke] = []
        out.reserveCapacity(maxCount)
        let last = strokes.count - 1
        for i in 0..<maxCount {
            let idx = maxCount == 1 ? 0 : Int(round(Double(i * last) / Double(maxCount - 1)))
            out.append(strokes[idx])
        }
        return out
    }
}
