import { expect, test, type Page } from '@playwright/test'

const CLIP = '/clips/gold_candidates/cand_demo'

async function openClip(page: Page) {
  await page.goto(CLIP)
  await expect(page.getByRole('img', { name: 'Quadro 1 de cand_demo' })).toBeVisible()
}

test.describe('Bancada de anotação', () => {
  test.describe.configure({ mode: 'serial' })

  test('nomeia os robôs, aprova o quadro e mantém tudo depois de recarregar', async ({ page }) => {
    await openClip(page)
    const inspector = page.getByRole('region', { name: 'Inspetor' })
    await expect(inspector.getByText('Pendente')).toBeVisible()

    await page.keyboard.press('Digit1')
    await expect(inspector.getByText('Robô A')).toBeVisible()
    await expect(inspector.getByText('Robô B')).toBeVisible()

    await page.keyboard.press('Enter')
    await expect(page.getByRole('img', { name: 'Quadro 2 de cand_demo' })).toBeVisible()

    await page.keyboard.press('ArrowLeft')
    await expect(page.getByRole('status')).toHaveText('salvo')

    await page.reload()
    await expect(inspector.getByText('Aprovado')).toBeVisible()
    await expect(inspector.getByText('Robô A')).toBeVisible()
  })

  test('propõe a identidade do quadro anterior ao avançar', async ({ page }) => {
    await openClip(page)
    await page.keyboard.press('ArrowRight')

    const inspector = page.getByRole('region', { name: 'Inspetor' })
    await expect(page.getByRole('img', { name: 'Quadro 2 de cand_demo' })).toBeVisible()
    await expect(inspector.getByText('Robô A')).toBeVisible()
    await expect(inspector.getByText('Robô B')).toBeVisible()
  })

  test('marca um evento e um round pela linha do tempo', async ({ page }) => {
    await openClip(page)
    const inspector = page.getByRole('region', { name: 'Inspetor' })

    await page.keyboard.press('KeyH')
    await expect(inspector.getByRole('button', { name: /Remover Início/ })).toBeVisible()

    await page.keyboard.press('KeyR')
    await page.keyboard.press('ArrowRight')
    await page.keyboard.press('ArrowRight')
    await page.keyboard.press('KeyR')
    await expect(inspector.getByRole('button', { name: 'Remover round 1' })).toBeVisible()
  })

  test('apaga a caixa selecionada e segue para a que sobra', async ({ page }) => {
    await openClip(page)
    await page.keyboard.press('ArrowRight')
    await page.keyboard.press('ArrowRight')
    await page.keyboard.press('ArrowRight')

    const inspector = page.getByRole('region', { name: 'Inspetor' })
    await expect(inspector.getByRole('button', { name: 'Apagar caixa 2' })).toBeVisible()
    await page.keyboard.press('Delete')

    await expect(inspector.getByRole('button', { name: 'Apagar caixa 2' })).toHaveCount(0)
    await expect(inspector.getByRole('button', { name: 'Apagar caixa 1' })).toBeVisible()
  })

  test('explica por que um clip com quadros pendentes não vira treino', async ({ page }) => {
    await openClip(page)
    await page.getByRole('button', { name: 'Mover', exact: true }).click()

    const dialog = page.getByRole('dialog')
    await expect(dialog.getByText(/quadros ainda estão pendentes de revisão/).first()).toBeVisible()
  })
})
