# bagrutai

BagrutAi – a self-contained chat UI built as a single HTML file (`bagrutai.html`).  
The interface is in **demo mode** (placeholder replies) until a model backend is connected.

## Serve locally

```bash
# Python 3
python -m http.server 8080
# then open http://localhost:8080/bagrutai.html
```

Any static file server works (e.g. `npx serve .`, `ruby -run -e httpd . -p 8080`).

## Embed in a site

```html
<iframe
  src="bagrutai.html"
  width="100%"
  height="700"
  style="border:none;border-radius:18px;"
  title="BagrutAi Chat"
></iframe>
```

## Features

- Message bubbles (user right, assistant left)
- **Enter** to send · **Shift + Enter** for a new line
- Textarea grows automatically (up to ~5 lines)
- **Clear** button resets the conversation
- Fully responsive (mobile-friendly)
- Zero dependencies – inline CSS & JS, no build step required

## Connect a model

Replace the demo block in `bagrutai.html` (search for `// UI-only demo response`) with a
`fetch` call to your API endpoint:

```js
const res = await fetch('/api/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ messages: conversation })
});
const { reply } = await res.json();
conversation.push({ role: 'assistant', content: reply });
addMessage('assistant', reply);
```
