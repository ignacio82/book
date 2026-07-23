const decoder = new TextDecoder();
let content = "";
const buf = new Uint8Array(1024);
while (true) {
  const n = Deno.stdin.readSync(buf);
  if (n === null || n === 0) break;
  content += decoder.decode(buf.subarray(0, n));
}
console.log(JSON.stringify({ files: [{ name: "app.R", content: content, type: "text" }] }));
