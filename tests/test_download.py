from rag import config, download

SAMPLES = config.RAW / "samples"


def test_valid_html():
    for name in ("1508.07909v5.html", "2412.09871v1.html"):
        assert download.html_is_valid((SAMPLES / name).read_text()) == (True, "ok")


def test_invalid_ar5iv_page():
    ok, reason = download.html_is_valid((SAMPLES / "2305.15425v2.ar5iv.html").read_text())
    print(reason)
    assert not ok and "No content available" in reason


def test_license():
    for name, want in [("1508.07909v5.html", "CC BY 4.0"),
                       ("2402.18376v2.html", "CC BY-NC-SA 4.0"),
                       ("2004.03720v2.html", "arXiv.org perpetual non-exclusive license")]:
        got = download.html_license((SAMPLES / name).read_text())
        print(name, got)
        assert got == want


def test_per_host_delay(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(download.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(download.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s))
    monkeypatch.setattr(download, "last_request", {})
    starts = []
    for host in ["a", "a", "b", "a"]:
        download.wait_for(host)
        starts.append((host, clock[0]))
        clock[0] += 0.3  # the request itself takes 0.3 s
    a_times = [t for h, t in starts if h == "a"]
    assert all(t2 - t1 >= config.FILE_DELAY for t1, t2 in zip(a_times, a_times[1:]))
    assert starts[2] == ("b", starts[1][1] + 0.3)  # another host does not wait
