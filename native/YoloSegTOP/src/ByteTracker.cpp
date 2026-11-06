#include "ByteTracker.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>

namespace
{
using Vec8 = std::array<double, 8>;
using Mat8 = std::array<double, 64>;

const double kStdPos = 1.0 / 20.0;
const double kStdVel = 1.0 / 160.0;

// Filtro de Kalman en espacio (cx, cy, aspecto, alto) - KalmanFilterXYAH de Ultralytics
struct Kalman
{
	static void initiate(const std::array<double, 4>& z, Vec8& mean, Mat8& cov)
	{
		mean = { z[0], z[1], z[2], z[3], 0, 0, 0, 0 };
		double h = z[3];
		double std[8] = { 2 * kStdPos * h, 2 * kStdPos * h, 1e-2, 2 * kStdPos * h,
						  10 * kStdVel * h, 10 * kStdVel * h, 1e-5, 10 * kStdVel * h };
		cov.fill(0);
		for (int i = 0; i < 8; i++)
			cov[i * 8 + i] = std[i] * std[i];
	}

	static void predict(Vec8& mean, Mat8& cov)
	{
		double h = mean[3];
		double std[8] = { kStdPos * h, kStdPos * h, 1e-2, kStdPos * h,
						  kStdVel * h, kStdVel * h, 1e-5, kStdVel * h };
		for (int i = 0; i < 4; i++)
			mean[i] += mean[i + 4];
		// cov = F cov F^T + Q,  F = [[I, I], [0, I]]
		Mat8 t;
		for (int r = 0; r < 8; r++)
			for (int c = 0; c < 8; c++)
				t[r * 8 + c] = cov[r * 8 + c] + (r < 4 ? cov[(r + 4) * 8 + c] : 0.0);
		for (int r = 0; r < 8; r++)
			for (int c = 0; c < 8; c++)
				cov[r * 8 + c] = t[r * 8 + c] + (c < 4 ? t[r * 8 + c + 4] : 0.0);
		for (int i = 0; i < 8; i++)
			cov[i * 8 + i] += std[i] * std[i];
	}

	static void update(Vec8& mean, Mat8& cov, const std::array<double, 4>& z)
	{
		double h = mean[3];
		double std[4] = { kStdPos * h, kStdPos * h, 1e-1, kStdPos * h };
		// S = H cov H^T + R (4x4)
		double S[16];
		for (int r = 0; r < 4; r++)
			for (int c = 0; c < 4; c++)
				S[r * 4 + c] = cov[r * 8 + c] + (r == c ? std[r] * std[r] : 0.0);
		double Si[16];
		invert4(S, Si);
		// K = cov H^T S^-1  (8x4)
		double K[32];
		for (int r = 0; r < 8; r++)
			for (int c = 0; c < 4; c++)
			{
				double s = 0;
				for (int k = 0; k < 4; k++)
					s += cov[r * 8 + k] * Si[k * 4 + c];
				K[r * 4 + c] = s;
			}
		double innov[4];
		for (int i = 0; i < 4; i++)
			innov[i] = z[i] - mean[i];
		for (int r = 0; r < 8; r++)
			for (int k = 0; k < 4; k++)
				mean[r] += K[r * 4 + k] * innov[k];
		// cov = cov - K S K^T  ==  cov - K (H cov)
		Mat8 n = cov;
		for (int r = 0; r < 8; r++)
			for (int c = 0; c < 8; c++)
			{
				double s = 0;
				for (int k = 0; k < 4; k++)
					s += K[r * 4 + k] * cov[k * 8 + c];
				n[r * 8 + c] -= s;
			}
		cov = n;
	}

	static void invert4(const double* m, double* inv)
	{
		// Gauss-Jordan 4x4 (S es simetrica definida positiva)
		double a[4][8];
		for (int r = 0; r < 4; r++)
			for (int c = 0; c < 8; c++)
				a[r][c] = c < 4 ? m[r * 4 + c] : (c - 4 == r ? 1.0 : 0.0);
		for (int col = 0; col < 4; col++)
		{
			int piv = col;
			for (int r = col + 1; r < 4; r++)
				if (std::fabs(a[r][col]) > std::fabs(a[piv][col]))
					piv = r;
			for (int c = 0; c < 8; c++)
				std::swap(a[col][c], a[piv][c]);
			double d = a[col][col];
			if (std::fabs(d) < 1e-12)
				d = 1e-12;
			for (int c = 0; c < 8; c++)
				a[col][c] /= d;
			for (int r = 0; r < 4; r++)
				if (r != col)
				{
					double f = a[r][col];
					for (int c = 0; c < 8; c++)
						a[r][c] -= f * a[col][c];
				}
		}
		for (int r = 0; r < 4; r++)
			for (int c = 0; c < 4; c++)
				inv[r * 4 + c] = a[r][c + 4];
	}
};

enum class State { New, Tracked, Lost, Removed };

std::array<double, 4> toXYAH(float x1, float y1, float x2, float y2)
{
	double w = x2 - x1, h = y2 - y1;
	return { x1 + w / 2, y1 + h / 2, w / std::max(h, 1e-6), h };
}

// Asignacion lineal con costo limite (equivale a lap.lapjv(extend_cost=True, cost_limit=thresh))
void linearAssignment(const std::vector<std::vector<double>>& cost, int rows, int cols, double thresh,
					  std::vector<std::pair<int, int>>& matches, std::vector<int>& uRows, std::vector<int>& uCols)
{
	matches.clear();
	uRows.clear();
	uCols.clear();
	if (rows == 0 || cols == 0)
	{
		for (int i = 0; i < rows; i++) uRows.push_back(i);
		for (int j = 0; j < cols; j++) uCols.push_back(j);
		return;
	}
	// Matriz extendida (rows+cols)^2: asignar a un "dummy" cuesta thresh/2
	const int n = rows + cols;
	std::vector<double> a((n + 1) * (n + 1), 0.0);
	auto A = [&](int i, int j) -> double& { return a[i * (n + 1) + j]; };
	for (int i = 1; i <= n; i++)
		for (int j = 1; j <= n; j++)
		{
			bool realRow = i <= rows, realCol = j <= cols;
			A(i, j) = (realRow && realCol) ? cost[i - 1][j - 1] : (!realRow && !realCol ? 0.0 : thresh / 2.0);
		}
	// Hungaro O(n^3) con potenciales
	const double INF = std::numeric_limits<double>::max() / 4;
	std::vector<double> u(n + 1, 0), v(n + 1, 0);
	std::vector<int> p(n + 1, 0), way(n + 1, 0);
	for (int i = 1; i <= n; i++)
	{
		p[0] = i;
		int j0 = 0;
		std::vector<double> minv(n + 1, INF);
		std::vector<char> used(n + 1, 0);
		do
		{
			used[j0] = 1;
			int i0 = p[j0], j1 = 0;
			double delta = INF;
			for (int j = 1; j <= n; j++)
				if (!used[j])
				{
					double cur = A(i0, j) - u[i0] - v[j];
					if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
					if (minv[j] < delta) { delta = minv[j]; j1 = j; }
				}
			for (int j = 0; j <= n; j++)
				if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
				else minv[j] -= delta;
			j0 = j1;
		} while (p[j0] != 0);
		do
		{
			int j1 = way[j0];
			p[j0] = p[j1];
			j0 = j1;
		} while (j0);
	}
	std::vector<int> rowToCol(rows, -1), colToRow(cols, -1);
	for (int j = 1; j <= cols; j++)
		if (p[j] >= 1 && p[j] <= rows)
		{
			int i = p[j] - 1;
			if (cost[i][j - 1] <= thresh)
			{
				rowToCol[i] = j - 1;
				colToRow[j - 1] = i;
			}
		}
	for (int i = 0; i < rows; i++)
		if (rowToCol[i] >= 0) matches.push_back({ i, rowToCol[i] });
		else uRows.push_back(i);
	for (int j = 0; j < cols; j++)
		if (colToRow[j] < 0) uCols.push_back(j);
}

double iou(const std::array<float, 4>& a, const std::array<float, 4>& b)
{
	float iw = std::min(a[2], b[2]) - std::max(a[0], b[0]);
	float ih = std::min(a[3], b[3]) - std::max(a[1], b[1]);
	if (iw <= 0 || ih <= 0)
		return 0.0;
	double inter = (double)iw * ih;
	double ua = (double)(a[2] - a[0]) * (a[3] - a[1]) + (double)(b[2] - b[0]) * (b[3] - b[1]) - inter;
	return inter / (ua + 1e-7);
}
} // namespace

struct ByteTracker::Track
{
	std::array<float, 4> detBox;   // caja de la deteccion (antes de tener Kalman)
	Vec8 mean{};
	Mat8 cov{};
	bool hasKalman = false;
	float score = 0;
	int idx = -1;
	int trackId = 0;
	bool isActivated = false;
	State state = State::New;
	int frameId = 0;
	int startFrame = 0;

	std::array<float, 4> box() const
	{
		if (!hasKalman)
			return detBox;
		double w = mean[2] * mean[3], h = mean[3];
		return { (float)(mean[0] - w / 2), (float)(mean[1] - h / 2), (float)(mean[0] + w / 2), (float)(mean[1] + h / 2) };
	}
	std::array<double, 4> xyah() const { return toXYAH(detBox[0], detBox[1], detBox[2], detBox[3]); }

	void activate(int& nextId, int frame)
	{
		trackId = ++nextId;
		Kalman::initiate(xyah(), mean, cov);
		hasKalman = true;
		state = State::Tracked;
		isActivated = (frame == 1);
		frameId = frame;
		startFrame = frame;
	}
	void reActivate(const Track& det, int frame)
	{
		Kalman::update(mean, cov, det.xyah());
		state = State::Tracked;
		isActivated = true;
		frameId = frame;
		score = det.score;
		idx = det.idx;
	}
	void update(const Track& det, int frame)
	{
		frameId = frame;
		Kalman::update(mean, cov, det.xyah());
		state = State::Tracked;
		isActivated = true;
		score = det.score;
		idx = det.idx;
	}
};

namespace
{
using TrackPtr = ByteTracker::TrackPtr;

std::vector<std::vector<double>> iouDistance(const std::vector<TrackPtr>& a, const std::vector<TrackPtr>& b)
{
	std::vector<std::vector<double>> d(a.size(), std::vector<double>(b.size(), 1.0));
	for (size_t i = 0; i < a.size(); i++)
	{
		auto ba = a[i]->box();
		for (size_t j = 0; j < b.size(); j++)
			d[i][j] = 1.0 - iou(ba, b[j]->box());
	}
	return d;
}

std::vector<std::vector<double>> getDists(const std::vector<TrackPtr>& tracks, const std::vector<TrackPtr>& dets, bool fuse)
{
	auto d = iouDistance(tracks, dets);
	if (fuse)
		for (auto& row : d)
			for (size_t j = 0; j < row.size(); j++)
				row[j] = 1.0 - (1.0 - row[j]) * dets[j]->score;
	return d;
}

std::vector<TrackPtr> joint(const std::vector<TrackPtr>& a, const std::vector<TrackPtr>& b)
{
	std::vector<TrackPtr> res = a;
	std::unordered_set<int> ids;
	for (auto& t : a) ids.insert(t->trackId);
	for (auto& t : b)
		if (ids.insert(t->trackId).second)
			res.push_back(t);
	return res;
}

std::vector<TrackPtr> sub(const std::vector<TrackPtr>& a, const std::vector<TrackPtr>& b)
{
	std::unordered_set<int> ids;
	for (auto& t : b) ids.insert(t->trackId);
	std::vector<TrackPtr> res;
	for (auto& t : a)
		if (!ids.count(t->trackId))
			res.push_back(t);
	return res;
}
} // namespace

std::vector<TrackResult> ByteTracker::untracked(const std::vector<Detection>& dets, float thresh)
{
	std::vector<TrackResult> out;
	for (const auto& d : dets)
		if (d.score >= thresh)
			out.push_back({ 0, d.x1, d.y1, d.x2, d.y2, d.score, d.index });
	std::stable_sort(out.begin(), out.end(), [](const TrackResult& a, const TrackResult& b) { return a.score > b.score; });
	for (size_t i = 0; i < out.size(); i++)
		out[i].id = (int)i + 1;
	return out;
}

void ByteTracker::reset()
{
	myTracked.clear();
	myLost.clear();
	myFrameId = 0;
	myNextId = 0;
}

std::vector<TrackResult> ByteTracker::update(const std::vector<Detection>& dets, const ByteTrackParams& p)
{
	myFrameId++;
	const int frame = myFrameId;
	std::vector<TrackPtr> activated, refind, lost, removed;

	std::vector<TrackPtr> detsHigh, detsLow;
	for (const auto& d : dets)
	{
		auto t = std::make_shared<Track>();
		t->detBox = { d.x1, d.y1, d.x2, d.y2 };
		t->score = d.score;
		t->idx = d.index;
		if (d.score >= p.highThresh)
			detsHigh.push_back(t);
		else if (d.score > p.lowThresh)
			detsLow.push_back(t);
	}

	std::vector<TrackPtr> unconfirmed, trackedStracks;
	for (auto& t : myTracked)
		(t->isActivated ? trackedStracks : unconfirmed).push_back(t);

	// 1) asociacion con detecciones de alta confianza
	auto pool = joint(trackedStracks, myLost);
	for (auto& t : pool)
	{
		if (t->state != State::Tracked)
			t->mean[7] = 0;
		Kalman::predict(t->mean, t->cov);
	}
	std::vector<std::pair<int, int>> matches;
	std::vector<int> uTrack, uDet, uDummy;
	linearAssignment(getDists(pool, detsHigh, p.fuseScore), (int)pool.size(), (int)detsHigh.size(), p.matchThresh, matches, uTrack, uDet);
	for (auto& m : matches)
	{
		auto& t = pool[m.first];
		auto& d = detsHigh[m.second];
		if (t->state == State::Tracked) { t->update(*d, frame); activated.push_back(t); }
		else { t->reActivate(*d, frame); refind.push_back(t); }
	}

	// 2) tracks sin match contra detecciones de baja confianza
	std::vector<TrackPtr> rTracked;
	for (int i : uTrack)
		if (pool[i]->state == State::Tracked)
			rTracked.push_back(pool[i]);
	std::vector<int> uTrack2;
	linearAssignment(iouDistance(rTracked, detsLow), (int)rTracked.size(), (int)detsLow.size(), 0.5, matches, uTrack2, uDummy);
	for (auto& m : matches)
	{
		auto& t = rTracked[m.first];
		auto& d = detsLow[m.second];
		if (t->state == State::Tracked) { t->update(*d, frame); activated.push_back(t); }
		else { t->reActivate(*d, frame); refind.push_back(t); }
	}
	for (int i : uTrack2)
	{
		auto& t = rTracked[i];
		if (t->state != State::Lost) { t->state = State::Lost; lost.push_back(t); }
	}

	// 3) tracks sin confirmar (vistos un solo frame)
	std::vector<TrackPtr> remaining;
	for (int i : uDet) remaining.push_back(detsHigh[i]);
	std::vector<int> uUnconf, uDet3;
	linearAssignment(getDists(unconfirmed, remaining, p.fuseScore), (int)unconfirmed.size(), (int)remaining.size(), 0.7, matches, uUnconf, uDet3);
	for (auto& m : matches)
	{
		unconfirmed[m.first]->update(*remaining[m.second], frame);
		activated.push_back(unconfirmed[m.first]);
	}
	for (int i : uUnconf)
	{
		unconfirmed[i]->state = State::Removed;
		removed.push_back(unconfirmed[i]);
	}

	// 4) tracks nuevos
	for (int i : uDet3)
	{
		auto& t = remaining[i];
		if (t->score < p.newTrackThresh)
			continue;
		t->activate(myNextId, frame);
		activated.push_back(t);
	}

	// 5) borrar perdidos viejos
	for (auto& t : myLost)
		if (frame - t->frameId > p.trackBuffer)
		{
			t->state = State::Removed;
			removed.push_back(t);
		}

	std::vector<TrackPtr> tracked;
	for (auto& t : myTracked)
		if (t->state == State::Tracked)
			tracked.push_back(t);
	tracked = joint(tracked, activated);
	tracked = joint(tracked, refind);
	auto lostAll = sub(myLost, tracked);
	lostAll.insert(lostAll.end(), lost.begin(), lost.end());
	lostAll = sub(lostAll, removed);

	// quitar duplicados entre tracked y lost (se queda el mas viejo)
	auto pd = iouDistance(tracked, lostAll);
	std::unordered_set<int> dupA, dupB;
	for (size_t i = 0; i < tracked.size(); i++)
		for (size_t j = 0; j < lostAll.size(); j++)
			if (pd[i][j] < 0.15)
			{
				int tp = tracked[i]->frameId - tracked[i]->startFrame;
				int tq = lostAll[j]->frameId - lostAll[j]->startFrame;
				if (tp > tq) dupB.insert((int)j);
				else dupA.insert((int)i);
			}
	myTracked.clear();
	for (size_t i = 0; i < tracked.size(); i++)
		if (!dupA.count((int)i)) myTracked.push_back(tracked[i]);
	myLost.clear();
	for (size_t j = 0; j < lostAll.size(); j++)
		if (!dupB.count((int)j)) myLost.push_back(lostAll[j]);

	std::vector<TrackResult> out;
	for (auto& t : myTracked)
		if (t->isActivated)
		{
			auto b = t->box();
			out.push_back({ t->trackId, b[0], b[1], b[2], b[3], t->score, t->idx });
		}
	return out;
}
