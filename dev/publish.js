function toggleIndexPage() {
    var page = document.getElementById("content");
    var toc = document.getElementById("toc");
    var navigation = document.getElementById("page-navigation");
    // Get the current page state. Default to 'page'.
    var state = localStorage.getItem("pageState");
    var mobile = localStorage.getItem("mobile");
    // Toggle index based on 'state'.
    if (state === "page") {
        // Save the current scroll position on the page.
        localStorage.setItem("scrollTop", document.documentElement.scrollTop);
        // Hide the page and display the table-of-contents.
        toc.style.display = "block";
        if (mobile === "true") {
            page.style.display = "none";
            navigation.style.display = "none";
        }
        // Scroll to the top of the table of contents.
        document.documentElement.scrollTop = 0;
        localStorage.setItem("pageState", "index");
    } else if (state === "index") {
        // Hide the table-of-contents and display the page.
        toc.style.display = "none";
        page.style.display = "block";
        navigation.style.display = "flex";
        // Restore the saved position on the page.
        document.documentElement.scrollTop = localStorage.getItem("scrollTop");
        localStorage.setItem("pageState", "page");
    }
}

function getQueryVariable(variable) {
    var query = window.location.search.substring(1);
    var vars = query.split("&");
    for (var i=0; i<vars.length; i++) {
        var pair = vars[i].split("=");
        if (pair[0] == variable) {
            return pair[1];
        }
    }
    return null;
}

function triggerEvent(type, obj, data){
    var ev;
    if (document.createEvent) {
        ev = document.createEvent("HTMLEvents");
        ev.initEvent(type, true, true);
    } else {
        ev = document.createEventObject();
        ev.eventType = type;
    }
    ev.eventName = type;
    if (data) {
        for (var key in data) {
            if (data.hasOwnProperty(key)) {
                ev[key] = data[key];
            }
        }
    }
    if (document.createEvent) {
        obj.dispatchEvent(ev);
    } else {
        obj.fireEvent("on" + ev.eventType, ev);
    }
}

window.addEventListener("searchIndexLoaded", function (_) {
    var search = getQueryVariable("search");
    var div = document.getElementById("search-results");
    if (search !== null && div.innerHTML == "") {
        var search = decodeURIComponent(search);
        var results = window.searchIndex.search(search);
        var details = document.createElement("p");
        div.appendChild(details);
        details.innerHTML = "Results: " + results.length;
        results.forEach(function (item) {
            var p = document.createElement("p");
            div.appendChild(p);
            var a = document.createElement("a");
            a.href = item["ref"];
            a.innerText = item["ref"];
            p.appendChild(a);
        });
    }
});

var mql = window.matchMedia('screen and (max-width:760px)');
mql.addEventListener("change",
    function(mq) {
        var state = localStorage.getItem("pageState");
        if (mq.matches) {
            localStorage.setItem("mobile", true);
            if (state === "index") {
                toggleIndexPage();
            }
        } else {
            localStorage.setItem("mobile", false);
            if (state === "page") {
                toggleIndexPage();
            }
        }
    }
);

document.addEventListener("DOMContentLoaded", function () {
    if (window.matchMedia('(max-width:760px)').matches) {
        localStorage.setItem("mobile", true);
    } else {
        localStorage.setItem("mobile", false);
    }
    // Hide the navigation section.
    if (localStorage.getItem("mobile") === "true") {
        localStorage.setItem("pageState", "index");
        toggleIndexPage();
    } else {
        localStorage.setItem("pageState", "index");
    }
    // Tabulator init.
    var table = document.getElementById("docstring-index");
    if (table !== null) {
        var table = new Tabulator("#docstring-index", {
            layout: "fitColumns",
            persistentSort:true,
            persistenceMode:"local",
            columns: [
                {title: "Name", formatter: "html", headerFilter:true},
                {title: "Module", formatter: "html", headerFilter:true},
                {title: "Visibility", formatter: "html", headerFilter:true},
                {title: "Category", formatter: "html", headerFilter:true},
            ],
            initialSort: [
                {column: "name", dir: "asc"},
            ],
        });
    }
    // Version selector.
    if (PUBLISH_VERSIONS !== null) {
        var version_selector = document.querySelector("#version-selector");
        PUBLISH_VERSIONS.forEach(function (each) {
            var name = each[0];
            var path = each[1];
            version = document.createElement("option");
            version.value = path;
            version.text = name;
            version_selector.appendChild(version);
            // Set the current version.
            if (name == PUBLISH_VERSION) {
                version_selector.value = path;
            }
        });
        version_selector.addEventListener("change", function(_) {
            var url = version_selector.options[version_selector.selectedIndex].value;
            window.location.href = url;
        });
    }
    let searchBuilder = new lunr.Builder;
    searchBuilder.ref("id");
    searchBuilder.field("body");
    // Background loading of search index.
    var feedLoaded = function (e) {
        var feed = JSON.parse(e.target.response);
        feed.forEach(function (entry) {
            searchBuilder.add(entry);
        });
        window.searchIndex = searchBuilder.build();
        triggerEvent("searchIndexLoaded", window, {});
    }
    // Fire off the request for the JSON search data.
    if (getQueryVariable("search") !== null) {
        var xhr = new XMLHttpRequest;
        xhr.open('get', PUBLISH_ROOT + '/search.json');
        xhr.addEventListener("load", feedLoaded);
        xhr.send();
    }
    // Hook up the searchbar.
    var search_bar = document.getElementById("search-input");
    search_bar.addEventListener("change", function (event) {
        window.location.href = PUBLISH_ROOT + "/search.html?search=" + encodeURIComponent(event.target.value);
    });
});