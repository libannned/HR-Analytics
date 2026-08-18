"use client";

import { useMemo, useState } from "react";
import {
  Bell, Check, CheckCircle2, ChevronDown, ChevronRight, CircleHelp, Clock3,
  FileText, Filter, Inbox, LayoutGrid, Link2, Menu, MessageSquareText, MoreHorizontal,
  Plus, Search, Send, Settings, Sparkles, Users, Video, X, XCircle,
} from "lucide-react";

type Status = "Submitted" | "In review" | "Changes requested" | "Approved";
type Submission = {
  id: number; creator: string; handle: string; initials: string; campaign: string;
  brand: string; platform: string; title: string; time: string; status: Status;
  color: string; caption: string; version: number;
};

const initialSubmissions: Submission[] = [
  { id: 1, creator: "Maya Chen", handle: "@mayamakes", initials: "MC", campaign: "Summer Skin Reset", brand: "Nori Beauty", platform: "TikTok Shop", title: "Morning routine — final cut", time: "12 min ago", status: "Submitted", color: "#e8c6a8", caption: "My 3-step reset for calm, hydrated skin ✨ I've been using the Nori Barrier Set every morning for two weeks. #NoriPartner #SkinReset", version: 2 },
  { id: 2, creator: "Jordan Ellis", handle: "@jordane", initials: "JE", campaign: "Summer Skin Reset", brand: "Nori Beauty", platform: "Instagram Reels", title: "GRWM with Nori", time: "38 min ago", status: "In review", color: "#b8cad3", caption: "Come get ready with me while I share the products that brought my skin barrier back.", version: 1 },
  { id: 3, creator: "Amara Brooks", handle: "@amarab", initials: "AB", campaign: "Everyday Carry", brand: "Field Notes Co.", platform: "LTK", title: "What’s in my work tote", time: "1 hr ago", status: "Changes requested", color: "#cabdaf", caption: "Everything in my tote that makes a long studio day easier. Linking every detail below.", version: 1 },
  { id: 4, creator: "Theo Martin", handle: "@theomoves", initials: "TM", campaign: "Move Better", brand: "Aster Athletics", platform: "TikTok Shop", title: "5-minute mobility flow", time: "2 hrs ago", status: "Submitted", color: "#d5c2b4", caption: "Save this 5-minute flow for your next desk break. Wearing the Aster Form set.", version: 3 },
  { id: 5, creator: "Sofia Reyes", handle: "@sofiaedited", initials: "SR", campaign: "Everyday Carry", brand: "Field Notes Co.", platform: "ShopMy", title: "Desk essentials edit", time: "Yesterday", status: "Approved", color: "#aebdc0", caption: "A tightly edited list of the desk pieces I use every single day.", version: 1 },
];

const statusClass: Record<Status, string> = {
  "Submitted": "status submitted", "In review": "status review", "Changes requested": "status changes", "Approved": "status approved",
};

function PlatformMark({ name }: { name: string }) {
  const short = name === "TikTok Shop" ? "TT" : name === "Instagram Reels" ? "IG" : name === "ShopMy" ? "SM" : "LTK";
  return <span className={`platform-mark ${short.toLowerCase()}`}>{short}</span>;
}

export default function Page() {
  const [submissions, setSubmissions] = useState(initialSubmissions);
  const [selected, setSelected] = useState<Submission | null>(null);
  const [activeStatus, setActiveStatus] = useState("All");
  const [query, setQuery] = useState("");
  const [toast, setToast] = useState("");
  const [comment, setComment] = useState("");
  const [showCampaign, setShowCampaign] = useState(false);
  const [mobileNav, setMobileNav] = useState(false);

  const filtered = useMemo(() => submissions.filter(s => {
    const statusMatch = activeStatus === "All" || s.status === activeStatus;
    const q = query.toLowerCase();
    return statusMatch && (!q || `${s.creator} ${s.campaign} ${s.brand} ${s.title}`.toLowerCase().includes(q));
  }), [submissions, activeStatus, query]);

  const counts = {
    Submitted: submissions.filter(s => s.status === "Submitted").length,
    "In review": submissions.filter(s => s.status === "In review").length,
    "Changes requested": submissions.filter(s => s.status === "Changes requested").length,
    Approved: submissions.filter(s => s.status === "Approved").length,
  };

  function updateStatus(status: Status) {
    if (!selected) return;
    const updated = { ...selected, status };
    setSubmissions(items => items.map(item => item.id === selected.id ? updated : item));
    setSelected(updated);
    setToast(status === "Approved" ? `${selected.creator}'s draft is approved` : `Changes sent to ${selected.creator}`);
    setComment("");
    window.setTimeout(() => setToast(""), 3200);
  }

  return (
    <main className="app-shell">
      <aside className={`sidebar ${mobileNav ? "open" : ""}`}>
        <div className="brand"><span className="logo-mark"><Check size={17} strokeWidth={3}/></span><span>greenlight</span></div>
        <button className="workspace"><span className="workspace-avatar">NW</span><span><b>Northwind Social</b><small>Agency workspace</small></span><ChevronDown size={15}/></button>
        <nav>
          <p className="nav-label">Workspace</p>
          <a className="active"><Inbox size={18}/><span>Review inbox</span><em>{counts.Submitted + counts["In review"]}</em></a>
          <a><LayoutGrid size={18}/><span>Campaigns</span></a>
          <a><Users size={18}/><span>Creators</span></a>
          <p className="nav-label spaced">Manage</p>
          <a><FileText size={18}/><span>Activity</span></a>
          <a><Settings size={18}/><span>Settings</span></a>
        </nav>
        <div className="pilot-card"><Sparkles size={17}/><b>Growth plan trial</b><p>11 days left in your pilot.</p><span><i style={{width:"62%"}} /></span><button>View plan</button></div>
        <div className="user-card"><div className="avatar dark">AK</div><span><b>Alex Kim</b><small>alex@northwind.co</small></span><MoreHorizontal size={17}/></div>
      </aside>

      <section className="workspace-main">
        <header className="topbar">
          <button className="mobile-menu" onClick={() => setMobileNav(v => !v)}><Menu size={20}/></button>
          <div><p>Northwind Social <ChevronRight size={12}/> Review inbox</p><h1>Good morning, Alex</h1></div>
          <div className="top-actions"><button className="icon-button" aria-label="Help"><CircleHelp size={19}/></button><button className="icon-button has-dot" aria-label="Notifications"><Bell size={19}/></button><button className="primary" onClick={() => setShowCampaign(true)}><Plus size={17}/> New campaign</button></div>
        </header>

        <div className="content">
          <section className="intro-row"><div><h2>Review inbox</h2><p>Everything waiting on your team, across every platform.</p></div><div className="week-stat"><span><CheckCircle2 size={18}/></span><div><b>8 approved</b><small>this week</small></div><strong>+33%</strong></div></section>

          <section className="metric-grid">
            <button onClick={() => setActiveStatus("Submitted")} className={activeStatus === "Submitted" ? "selected" : ""}><span className="metric-icon amber"><Inbox size={18}/></span><div><small>Needs review</small><strong>{counts.Submitted}</strong><p><i className="dot amber-dot"/>2 added today</p></div></button>
            <button onClick={() => setActiveStatus("In review")} className={activeStatus === "In review" ? "selected" : ""}><span className="metric-icon blue"><Clock3 size={18}/></span><div><small>In review</small><strong>{counts["In review"]}</strong><p>With your team</p></div></button>
            <button onClick={() => setActiveStatus("Changes requested")} className={activeStatus === "Changes requested" ? "selected" : ""}><span className="metric-icon coral"><MessageSquareText size={18}/></span><div><small>Changes requested</small><strong>{counts["Changes requested"]}</strong><p>Waiting on creator</p></div></button>
            <button onClick={() => setActiveStatus("Approved")} className={activeStatus === "Approved" ? "selected" : ""}><span className="metric-icon green"><CheckCircle2 size={18}/></span><div><small>Approved</small><strong>{counts.Approved}</strong><p>Ready to post</p></div></button>
          </section>

          <section className="queue-card">
            <div className="queue-head"><div><h3>Content queue</h3><span>{filtered.length} submissions</span></div><div className="queue-tools"><label><Search size={17}/><input value={query} onChange={e => setQuery(e.target.value)} placeholder="Search creators or campaigns"/></label><button><Filter size={16}/> Filter</button></div></div>
            <div className="tabs">{["All","Submitted","In review","Changes requested","Approved"].map(tab => <button key={tab} onClick={() => setActiveStatus(tab)} className={activeStatus === tab ? "active" : ""}>{tab}{tab === "All" && <em>{submissions.length}</em>}</button>)}</div>
            <div className="table-wrap"><table><thead><tr><th>Creator</th><th>Content</th><th>Campaign</th><th>Status</th><th>Submitted</th><th></th></tr></thead><tbody>
              {filtered.map(item => <tr key={item.id} onClick={() => setSelected(item)}>
                <td><div className="creator-cell"><span className="avatar" style={{background:item.color}}>{item.initials}</span><div><b>{item.creator}</b><small>{item.handle}</small></div></div></td>
                <td><div className="content-cell"><span className="thumb" style={{background:`linear-gradient(145deg, ${item.color}, #665f58)`}}><Video size={16}/></span><div><b>{item.title}</b><small><PlatformMark name={item.platform}/>{item.platform} · v{item.version}</small></div></div></td>
                <td><b className="regular">{item.campaign}</b><small className="block">{item.brand}</small></td>
                <td><span className={statusClass[item.status]}><i/>{item.status}</span></td><td className="muted">{item.time}</td><td><button className="row-go" aria-label={`Review ${item.title}`}><ChevronRight size={18}/></button></td>
              </tr>)}
            </tbody></table>{filtered.length === 0 && <div className="empty"><CheckCircle2 size={30}/><b>You’re all caught up</b><p>No submissions match this view.</p></div>}</div>
          </section>
        </div>
      </section>

      {selected && <div className="overlay" onMouseDown={e => e.target === e.currentTarget && setSelected(null)}><aside className="review-drawer">
        <div className="drawer-head"><div><span className={statusClass[selected.status]}><i/>{selected.status}</span><h2>{selected.title}</h2><p>{selected.campaign} · {selected.brand}</p></div><button className="icon-button" onClick={() => setSelected(null)}><X size={20}/></button></div>
        <div className="drawer-scroll"><div className="video-preview" style={{background:`linear-gradient(155deg, ${selected.color}, #373533)`}}><span className="play"><Video size={25}/></span><small>Draft preview · 0:32</small><em>v{selected.version}</em></div>
          <div className="creator-summary"><span className="avatar" style={{background:selected.color}}>{selected.initials}</span><div><b>{selected.creator}</b><small>{selected.handle}</small></div><PlatformMark name={selected.platform}/><span className="platform-name">{selected.platform}</span></div>
          <section className="detail-section"><div className="section-title"><b>Caption</b><span>198 / 2,200</span></div><p className="caption-copy">{selected.caption}</p></section>
          <section className="detail-section"><div className="section-title"><b>Campaign brief</b><a>Open brief <ChevronRight size={14}/></a></div><div className="brief-check"><Check size={15}/><span>Show the product in the first 3 seconds</span></div><div className="brief-check"><Check size={15}/><span>Include #Partner disclosure</span></div></section>
          <section className="detail-section"><div className="section-title"><b>Review note</b><span>Visible to creator</span></div><textarea value={comment} onChange={e => setComment(e.target.value)} placeholder="Add clear, actionable feedback…"/><div className="quick-notes"><button onClick={() => setComment("Please show the product earlier in the video.")}>Product earlier</button><button onClick={() => setComment("Please add the required partnership disclosure.")}>Add disclosure</button></div></section>
        </div>
        <div className="drawer-actions"><button className="request" onClick={() => updateStatus("Changes requested")}><MessageSquareText size={17}/> Request changes</button><button className="approve" onClick={() => updateStatus("Approved")}><Check size={18}/> Approve content</button></div>
      </aside></div>}

      {showCampaign && <div className="overlay centered"><div className="modal"><button className="modal-close" onClick={() => setShowCampaign(false)}><X size={19}/></button><span className="modal-icon"><Plus size={22}/></span><h2>Create a campaign</h2><p>Set up a multi-platform approval workflow in a few steps.</p><label>Campaign name<input defaultValue="Fall launch" /></label><label>Brand<select defaultValue=""><option value="" disabled>Select a brand</option><option>Nori Beauty</option><option>Field Notes Co.</option><option>Aster Athletics</option></select></label><fieldset><legend>Platforms</legend>{["TikTok Shop","Instagram Reels","LTK","ShopMy"].map((p,i) => <label key={p} className="check-option"><input type="checkbox" defaultChecked={i<2}/><PlatformMark name={p}/>{p}</label>)}</fieldset><button className="primary wide" onClick={() => {setShowCampaign(false);setToast("Campaign draft created")}}><Plus size={17}/> Create campaign</button></div></div>}
      {toast && <div className="toast"><CheckCircle2 size={18}/>{toast}</div>}
    </main>
  );
}
