import Foundation

/// Exact Swift port of `LatexTokenizer.decode` and `detokenize_latex`.
public struct LatexTokenizer: Sendable {
    public let tokens: [String]
    public let padId: Int
    public let bosId: Int
    public let eosId: Int
    public let unkId: Int

    private static let specialTokens: Set<String> = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]

    public init(manifest: ModelManifest) {
        self.tokens = manifest.tokens
        self.padId = manifest.padId
        self.bosId = manifest.bosId
        self.eosId = manifest.eosId
        self.unkId = manifest.unkId
    }

    public func idsToTokens(_ ids: [Int]) -> [String] {
        var result: [String] = []
        result.reserveCapacity(ids.count)
        for id in ids {
            let token = (id >= 0 && id < tokens.count) ? tokens[id] : "<UNK>"
            if token == "<EOS>" {
                break
            }
            if Self.specialTokens.contains(token) {
                continue
            }
            result.append(token)
        }
        return result
    }

    public func decode(_ ids: [Int]) -> String {
        Self.detokenize(idsToTokens(ids))
    }

    public static func detokenize(_ tokens: [String]) -> String {
        var out = ""
        var previous: String? = nil
        for token in tokens {
            if let prev = previous, isControlWord(prev), let first = token.first, first.isLetter {
                out.append(" ")
            }
            out.append(token)
            previous = token
        }
        return out
    }

    /// Matches Python regex `\\[A-Za-z]+\Z`
    private static func isControlWord(_ token: String) -> Bool {
        guard token.count >= 2, token.first == "\\" else { return false }
        for scalar in token.dropFirst().unicodeScalars {
            let v = scalar.value
            let isAsciiLetter = (v >= 65 && v <= 90) || (v >= 97 && v <= 122)
            if !isAsciiLetter { return false }
        }
        return true
    }
}
