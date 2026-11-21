import SwiftUI
import WebKit

/// Offline KaTeX renderer backed by bundled `Resources/katex/` with smooth vertical & horizontal scrolling
/// so multi-line equations, matrices, and wide derivations never get clipped.
public struct KaTeXWebView: UIViewRepresentable {
    let latex: String
    let isDark: Bool

    public func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    public func makeUIView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        let webView = WKWebView(frame: .zero, configuration: config)
        webView.isOpaque = false
        webView.backgroundColor = .clear
        webView.scrollView.backgroundColor = .clear
        webView.scrollView.isScrollEnabled = true
        webView.scrollView.bounces = true
        webView.scrollView.showsVerticalScrollIndicator = true
        webView.scrollView.showsHorizontalScrollIndicator = true
        webView.navigationDelegate = context.coordinator

        let baseURL = Bundle.main.url(forResource: "katex", withExtension: nil)
            ?? Bundle.main.resourceURL

        let html = """
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="utf-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=2.0, user-scalable=yes" />
          <link rel="stylesheet" href="katex.min.css" />
          <style>
            :root { color-scheme: light dark; }
            html, body {
              margin: 0;
              padding: 0;
              min-width: 100%;
              min-height: 100%;
              background: transparent;
              display: flex;
              font-family: -apple-system, BlinkMacSystemFont, sans-serif;
              -webkit-overflow-scrolling: touch;
            }
            #math {
              margin: auto;
              font-size: 26px;
              padding: 12px 20px;
              color: #111827;
            }
            body.dark #math {
              color: #F9FAFB;
            }
            #math .katex-display {
              margin: 0.25em 0;
            }
            #placeholder {
              font-size: 16px;
              color: #6B7280;
            }
            .katex-error {
              font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
              font-size: 15px;
              color: #EF4444;
            }
          </style>
          <script src="katex.min.js"></script>
        </head>
        <body class="\(isDark ? "dark" : "")">
          <div id="math"><span id="placeholder">Write a math expression below…</span></div>
          <script>
            window.setTheme = function(dark) {
              document.body.className = dark ? 'dark' : '';
            };
            window.renderLatex = function(tex) {
              const el = document.getElementById('math');
              if (!tex || !tex.trim()) {
                el.innerHTML = '<span id="placeholder">Write a math expression below…</span>';
                window.scrollTo(0, 0);
                return;
              }
              if (typeof katex === 'undefined') {
                el.textContent = tex;
                return;
              }
              try {
                katex.render(tex, el, { displayMode: true, throwOnError: false, strict: false });
              } catch (e) {
                el.innerHTML = '<span class="katex-error">' + tex.replace(/</g, '&lt;') + '</span>';
              }
            };
          </script>
        </body>
        </html>
        """
        context.coordinator.pendingLatex = latex
        context.coordinator.pendingDark = isDark
        webView.loadHTMLString(html, baseURL: baseURL)
        return webView
    }

    public func updateUIView(_ uiView: WKWebView, context: Context) {
        context.coordinator.update(webView: uiView, latex: latex, isDark: isDark)
    }

    public final class Coordinator: NSObject, WKNavigationDelegate {
        var isLoaded = false
        var pendingLatex: String = ""
        var pendingDark: Bool = false
        private var lastLatex: String?
        private var lastDark: Bool?

        public func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            isLoaded = true
            apply(webView: webView, latex: pendingLatex, isDark: pendingDark, force: true)
        }

        func update(webView: WKWebView, latex: String, isDark: Bool) {
            pendingLatex = latex
            pendingDark = isDark
            guard isLoaded else { return }
            apply(webView: webView, latex: latex, isDark: isDark, force: false)
        }

        private func apply(webView: WKWebView, latex: String, isDark: Bool, force: Bool) {
            if force || lastDark != isDark {
                lastDark = isDark
                webView.evaluateJavaScript("window.setTheme(\(isDark ? "true" : "false"));", completionHandler: nil)
            }
            if force || lastLatex != latex {
                lastLatex = latex
                if let encoded = try? JSONEncoder().encode(latex),
                   let jsonLiteral = String(data: encoded, encoding: .utf8) {
                    webView.evaluateJavaScript("window.renderLatex(\(jsonLiteral));", completionHandler: nil)
                }
            }
        }
    }
}
