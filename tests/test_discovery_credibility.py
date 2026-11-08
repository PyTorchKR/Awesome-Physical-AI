import discovery_credibility as credibility


def test_detect_venue_requires_acceptance():
    assert credibility.detect_venue("Accepted to CoRL 2026", "") == "CoRL"
    assert credibility.detect_venue("", "IEEE Robotics and Automation Letters, 2026") == "RA-L"
    assert credibility.detect_venue("Submitted to ICRA 2027", "") == ""
    assert credibility.detect_venue("Under review at RSS", "") == ""
    assert credibility.detect_venue("8 pages, 4 figures", "") == ""


def test_match_institutions_from_author_block_and_official_namespace():
    found = credibility.match_institutions(
        "Jane Doe Affiliation: Carnegie Mellon University, Pittsburgh",
        ["https://github.com/NVlabs/policy", "https://github.com/someone/policy"],
    )

    assert found == ["NVIDIA", "CMU"]


def test_match_institutions_avoids_lookalike_names():
    text = (
        "Affiliation: Washington University in St. Louis; Shanghai AI Institute of Things; "
        "Mila Novak; Google DeepMind; meta-learning group"
    )

    assert credibility.match_institutions(text, []) == ["Google DeepMind"]


def test_extract_author_block_strips_markup():
    html = (
        '<div class="ltx_authors"><span class="ltx_creator">Ann</span>'
        '<span class="ltx_role_affiliation">Stanford &amp; NVIDIA</span></div>\n</div>'
    )

    assert credibility.extract_author_block(html) == "Ann Stanford & NVIDIA"


def test_extract_author_block_stops_before_abstract():
    html = (
        '<div class="ltx_authors"><span>Ann</span><span>New York University</span>'
        '<div class="ltx_abstract"><p>Toward physical intelligence on robots.</p></div></div>\n</div>'
    )

    block = credibility.extract_author_block(html)

    assert "physical intelligence" not in block
    assert credibility.match_institutions(block, []) == ["NYU"]
