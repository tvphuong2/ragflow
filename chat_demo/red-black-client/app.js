import { h, render } from 'https://cdn.jsdelivr.net/npm/preact@10.19.3/dist/preact.module.js';
import { useEffect, useMemo, useRef, useState } from 'https://cdn.jsdelivr.net/npm/preact@10.19.3/hooks/dist/hooks.module.js';
import htm from 'https://cdn.jsdelivr.net/npm/htm@3.1.1/dist/htm.module.js';

const html = htm.bind(h);

const API_BASE = '/v1';

const formatTimestamp = (value) => {
  if (!value) return '—';
  const numeric = Number(value);
  if (Number.isNaN(numeric)) {
    return value;
  }
  const ms = numeric > 1e12 ? numeric : numeric * 1000;
  try {
    const dt = new Date(ms);
    if (Number.isNaN(dt.getTime())) return '—';
    return dt.toLocaleString([], { hour: '2-digit', minute: '2-digit', year: 'numeric', month: 'short', day: 'numeric' });
  } catch (err) {
    return '—';
  }
};

const safeArray = (input) => {
  if (!input) return [];
  if (Array.isArray(input)) return input;
  if (typeof input === 'object') return Object.values(input);
  return [];
};

const normaliseReference = (reference) => {
  if (!reference || typeof reference !== 'object') return null;
  return {
    ...reference,
    chunks: safeArray(reference.chunks),
    doc_aggs: safeArray(reference.doc_aggs),
  };
};

const uuid = () => {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
};

const apiFetch = async (path, { method = 'GET', body, headers = {}, signal } = {}) => {
  const url = path.startsWith('http') ? path : `${API_BASE}${path}`;
  const config = {
    method,
    headers: {
      'Content-Type': 'application/json',
      ...headers,
    },
    credentials: 'include',
    signal,
  };

  if (body !== undefined) {
    config.body = typeof body === 'string' ? body : JSON.stringify(body);
  }

  const response = await fetch(url, config);
  let payload;
  try {
    payload = await response.json();
  } catch (error) {
    throw new Error('Máy chủ trả về dữ liệu không hợp lệ.');
  }

  if (!response.ok) {
    throw new Error(payload?.message || `Yêu cầu thất bại (${response.status})`);
  }

  if (typeof payload === 'object' && payload !== null && 'code' in payload && payload.code !== 0) {
    throw new Error(payload.message || 'Yêu cầu thất bại.');
  }

  return payload?.data ?? null;
};

const mergeConversationMessages = (conversation) => {
  if (!conversation) {
    return [];
  }
  const refs = safeArray(conversation.reference);
  let refIdx = 0;

  const messages = safeArray(conversation.message).map((msg) => {
    const enriched = { ...msg };
    if (!enriched.id) {
      enriched.id = uuid();
    }
    if (enriched.role === 'assistant') {
      const ref = refs[refIdx];
      refIdx += 1;
      const normalised = normaliseReference(ref);
      if (normalised) {
        enriched.reference = normalised;
      }
    }
    return enriched;
  });

  return messages;
};

const ChunkCitation = ({ chunk }) => {
  if (!chunk) return null;
  const positions = safeArray(chunk.positions)
    .map((pos) => (typeof pos === 'number' ? pos : null))
    .filter((pos) => pos !== null)
    .slice(0, 4)
    .join(', ');
  return html`<div class="chunk-card">
    <strong>${chunk.document_name || 'Tài liệu'}</strong>
    <div>${chunk.content || '—'}</div>
    ${positions
      ? html`<div class="chunk-meta">Vị trí: <span>${positions}${chunk.positions?.length > 4 ? ', …' : ''}</span></div>`
      : null}
  </div>`;
};

const CitationPanel = ({ reference }) => {
  if (!reference) return null;
  const docs = safeArray(reference.doc_aggs);
  const chunks = safeArray(reference.chunks);
  if (!docs.length && !chunks.length) return null;
  return html`<div class="citation-panel">
    <div class="citation-title">Trích dẫn</div>
    ${docs.length
      ? html`<ul class="citation-docs">
          ${docs.map((doc) =>
            html`<li>
              <span>${doc.doc_name || doc.doc_id || 'Nguồn'}</span>
              ${doc.count !== undefined
                ? html`<span class="similarity">${doc.count} đoạn</span>`
                : null}
            </li>`
          )}
        </ul>`
      : null}
    ${chunks.length ? html`<div class="citation-chunks">${chunks.map((chunk) => html`<${ChunkCitation} chunk=${chunk} />`)}</div>` : null}
  </div>`;
};

const MessageBubble = ({ message }) => {
  const isUser = message.role === 'user';
  const roleLabel = isUser ? 'Người dùng' : 'Trợ lý';
  const content = message.content || '';
  return html`<div class=${`message ${isUser ? 'user' : 'assistant'}`} data-role=${roleLabel}>
    ${content.split('\n\n').map((paragraph, index) =>
      html`<p key=${`${message.id}-${index}`}>
        ${paragraph.split('\n').map((line, lineIndex) =>
          html`${lineIndex ? html`<br />` : null}${line}`
        )}
      </p>`
    )}
    ${!isUser && message.reference ? html`<${CitationPanel} reference=${message.reference} />` : null}
  </div>`;
};

const Spinner = () => html`<div class="spinner"></div>`;

const ErrorBanner = ({ message, onClose }) =>
  html`<div class="error-banner">
    <span>⚠️ ${message}</span>
    <button type="button" onClick=${onClose}>✕</button>
  </div>`;

const DialogCard = ({ dialog, active, onSelect }) => {
  const knowledge = safeArray(dialog.kb_names).slice(0, 2);
  return html`<button
    class=${`dialog-card ${active ? 'active' : ''}`}
    type="button"
    onClick=${() => onSelect(dialog)}
  >
    <h3>${dialog.name || 'Chat Bot'}</h3>
    <div class="dialog-meta">
      <span>${dialog.description || 'Không có mô tả'}</span>
    </div>
    ${knowledge.length
      ? html`<div class="dialog-meta">${knowledge.map((item) => html`<span class="tag">${item}</span>`)}</div>`
      : null}
  </button>`;
};

const ConversationCard = ({ conversation, active, onSelect }) => {
  const caption = conversation.update_time
    ? formatTimestamp(conversation.update_time)
    : formatTimestamp(conversation.create_time);
  return html`<button
    class=${`conversation-card ${active ? 'active' : ''}`}
    onClick=${() => onSelect(conversation)}
    type="button"
  >
    <h3>${conversation.name || 'Phiên chat'}</h3>
    <div class="conversation-meta">
      <span>${caption}</span>
      ${conversation.is_new ? html`<span class="status-pill">Mới</span>` : null}
    </div>
  </button>`;
};

const App = () => {
  const [dialogs, setDialogs] = useState([]);
  const [conversations, setConversations] = useState([]);
  const [messages, setMessages] = useState([]);
  const [selectedDialogId, setSelectedDialogId] = useState(null);
  const [selectedConversationId, setSelectedConversationId] = useState(null);
  const [dialogLoading, setDialogLoading] = useState(false);
  const [conversationLoading, setConversationLoading] = useState(false);
  const [messageLoading, setMessageLoading] = useState(false);
  const [composerValue, setComposerValue] = useState('');
  const [sending, setSending] = useState(false);
  const [error, setError] = useState(null);
  const chatScrollRef = useRef(null);

  const selectedDialog = useMemo(
    () => dialogs.find((dialog) => dialog.id === selectedDialogId) || null,
    [dialogs, selectedDialogId],
  );

  const selectedConversation = useMemo(
    () => conversations.find((conv) => conv.id === selectedConversationId) || null,
    [conversations, selectedConversationId],
  );

  const scrollToBottom = () => {
    if (chatScrollRef.current) {
      chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight;
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages.length]);

  useEffect(() => {
    const loadDialogs = async () => {
      setDialogLoading(true);
      try {
        const list = await apiFetch('/dialog/list');
        setDialogs(list || []);
        if (!selectedDialogId && list?.length) {
          setSelectedDialogId(list[0].id);
        }
      } catch (err) {
        setError(err.message);
      } finally {
        setDialogLoading(false);
      }
    };
    loadDialogs();
  }, []);

  useEffect(() => {
    if (!selectedDialogId) {
      setConversations([]);
      setSelectedConversationId(null);
      return;
    }
    const controller = new AbortController();
    const loadConversations = async () => {
      setConversationLoading(true);
      try {
        const query = new URLSearchParams({ dialog_id: selectedDialogId });
        const list = await apiFetch(`/conversation/list?${query.toString()}`, { signal: controller.signal });
        setConversations(list || []);
        if (!selectedConversationId && list?.length) {
          setSelectedConversationId(list[0].id);
        } else if (selectedConversationId && list && !list.some((conv) => conv.id === selectedConversationId)) {
          setSelectedConversationId(list[0]?.id ?? null);
        }
      } catch (err) {
        if (err.name !== 'AbortError') {
          setError(err.message);
        }
      } finally {
        setConversationLoading(false);
      }
    };
    loadConversations();
    return () => controller.abort();
  }, [selectedDialogId]);

  const applyConversationDetail = (conversation) => {
    setMessages(mergeConversationMessages(conversation));
  };

  useEffect(() => {
    if (!selectedConversationId) {
      setMessages([]);
      return;
    }
    const controller = new AbortController();
    const loadConversationDetail = async () => {
      setMessageLoading(true);
      try {
        const query = new URLSearchParams({ conversation_id: selectedConversationId });
        const detail = await apiFetch(`/conversation/get?${query.toString()}`, { signal: controller.signal });
        applyConversationDetail(detail);
      } catch (err) {
        if (err.name !== 'AbortError') {
          setError(err.message);
        }
      } finally {
        setMessageLoading(false);
      }
    };
    loadConversationDetail();
    return () => controller.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedConversationId]);

  const handleSelectDialog = (dialog) => {
    setSelectedDialogId(dialog.id);
    setSelectedConversationId(null);
    setMessages([]);
  };

  const handleSelectConversation = (conversation) => {
    if (!conversation) return;
    setSelectedConversationId(conversation.id);
  };

  const handleCreateConversation = async () => {
    if (!selectedDialogId || sending) return;
    setSending(true);
    try {
      const conversationId = uuid();
      const payload = {
        is_new: true,
        conversation_id: conversationId,
        dialog_id: selectedDialogId,
        name: `Phiên ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`,
      };
      const conversation = await apiFetch('/conversation/set', { method: 'POST', body: payload });
      await new Promise((resolve) => setTimeout(resolve, 100));
      setConversations((prev) => [{ id: conversation.id, name: conversation.name, create_time: Date.now(), is_new: true }, ...prev.filter((conv) => conv.id !== conversation.id)]);
      setSelectedConversationId(conversation.id);
      applyConversationDetail(conversation);
    } catch (err) {
      setError(err.message);
    } finally {
      setSending(false);
    }
  };

  const handleSend = async (event) => {
    event.preventDefault();
    if (!composerValue.trim() || !selectedConversationId || sending) return;
    const text = composerValue.trim();
    const userMessage = { id: uuid(), role: 'user', content: text };
    const outboundMessages = [...messages.map(({ role, content, id: mid }) => ({ role, content, id: mid })) , userMessage];
    setMessages((prev) => [...prev, userMessage]);
    setComposerValue('');
    setSending(true);
    try {
      const answer = await apiFetch('/conversation/completion', {
        method: 'POST',
        body: {
          conversation_id: selectedConversationId,
          messages: outboundMessages,
          stream: false,
        },
      });
      const assistantMessage = {
        id: answer?.id || uuid(),
        role: 'assistant',
        content: answer?.answer || '',
        reference: normaliseReference(answer?.reference),
      };
      setMessages((prev) => [...prev, assistantMessage]);
      setConversations((prev) =>
        prev.map((conv) =>
          conv.id === selectedConversationId
            ? { ...conv, update_time: Date.now(), is_new: false }
            : conv,
        ),
      );
      requestAnimationFrame(scrollToBottom);
    } catch (err) {
      setError(err.message);
      setMessages((prev) => prev.filter((msg) => msg.id !== userMessage.id));
      setComposerValue(text);
    } finally {
      setSending(false);
    }
  };

  const handleComposerChange = (event) => {
    setComposerValue(event.target.value);
  };

  return html`<div class="app-shell">
    ${error ? html`<${ErrorBanner} message=${error} onClose=${() => setError(null)} />` : null}
    <aside class="sidebar">
      <div class="sidebar-header">
        <h1>
          <span>RAGFlow</span>
          <span class="badge">Chat</span>
        </h1>
        <p>Chọn ứng dụng trò chuyện mà bạn muốn sử dụng.</p>
      </div>
      <div class="sidebar-list">
        ${dialogLoading
          ? html`<${Spinner} />`
          : dialogs.length
            ? dialogs.map((dialog) =>
                html`<${DialogCard}
                  key=${dialog.id}
                  dialog=${dialog}
                  active=${dialog.id === selectedDialogId}
                  onSelect=${handleSelectDialog}
                />`
              )
            : html`<div class="session-list-empty">Chưa có ứng dụng chat nào. Hãy tạo trong trang quản trị.</div>`}
      </div>
    </aside>
    <section class="session-column">
      <div class="session-header">
        <div>
          <h2>Phiên trò chuyện</h2>
          <p>Quản lý lịch sử trao đổi trong ứng dụng đã chọn.</p>
        </div>
        <div class="session-header-actions">
          <button class="button" type="button" onClick=${handleCreateConversation} disabled=${!selectedDialogId || sending}>
            + Phiên mới
          </button>
        </div>
      </div>
      <div class="session-list">
        ${conversationLoading
          ? html`<${Spinner} />`
          : conversations.length
            ? conversations.map((conversation) =>
                html`<${ConversationCard}
                  key=${conversation.id}
                  conversation=${conversation}
                  active=${conversation.id === selectedConversationId}
                  onSelect=${handleSelectConversation}
                />`
              )
            : html`<div class="session-list-empty">Chưa có phiên trò chuyện nào cho ứng dụng này.</div>`}
      </div>
    </section>
    <section class="chat-column">
      <div class="chat-header">
        <div class="chat-summary">
          <h2>${selectedConversation?.name || selectedDialog?.name || 'Chọn phiên chat'}</h2>
          <p>
            ${selectedDialog
              ? selectedDialog.description || 'Bắt đầu trò chuyện với mô hình AI tùy chỉnh của bạn.'
              : 'Hãy chọn một ứng dụng chat ở cột bên trái để bắt đầu.'}
          </p>
        </div>
        <div class="chat-stats">
          ${sending ? html`<span class="status-pill">Đang gửi…</span>` : null}
          ${messageLoading ? html`<span class="status-pill">Đang tải</span>` : null}
        </div>
      </div>
      <div class="message-scroll" ref=${chatScrollRef}>
        ${messageLoading
          ? html`<${Spinner} />`
          : messages.length
            ? html`<div class="message-group">
                ${messages.map((message) => html`<${MessageBubble} key=${message.id} message=${message} />`)}
              </div>`
            : html`<div class="message-empty">
                ${selectedConversationId
                  ? 'Chưa có nội dung nào trong phiên này. Hãy gửi tin nhắn đầu tiên của bạn.'
                  : 'Hãy tạo hoặc chọn một phiên trò chuyện để bắt đầu trao đổi.'}
              </div>`}
      </div>
      <form class="composer" onSubmit=${handleSend}>
        <textarea
          placeholder="Nhập câu hỏi của bạn bằng tiếng Việt..."
          value=${composerValue}
          onInput=${handleComposerChange}
          disabled=${!selectedConversationId || sending}
        ></textarea>
        <button class="button" type="submit" disabled=${!selectedConversationId || !composerValue.trim() || sending}>
          Gửi
        </button>
      </form>
    </section>
  </div>`;
};

render(html`<${App} />`, document.getElementById('app'));
